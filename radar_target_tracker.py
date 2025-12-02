import os
import numpy as np
import torch
import cv2
from typing import Dict, Any, List, Tuple, Optional, Iterable

from models.yoloRTv1.yolort import YOLORTv1
from deep_sort.deep_sort import DeepSort

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None


class RadarTargetTracker:
    """
    Radar 目标检测 + DeepSORT 跟踪封装。

    检测输出: [N,6] => [x1, y1, x2, y2, score, cls]
    跟踪输出: [M,6] => [x1, y1, x2, y2, cls, track_id]

    典型用法:
        tracker = RadarTargetTracker(cfg_model=..., num_classes=1,
                                     weight_path='weights/best_epoch_35_map_0.4619.pth')
        dets = tracker.detect(img)      # 单帧检测
        tracks = tracker.track(img)     # 单帧跟踪
        for tracks in tracker.track_sequence(img_list): ...
    """
    def __init__(
        self,
        cfg_model: Dict[str, Any],
        num_classes: int,
        device: str = "cuda",
        weight_path: Optional[str] = None,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.5,
        img_size: Tuple[int, int] = (512, 512),
        # DeepSORT
        deepsort_model_path: Optional[str] = None,
        deepsort_model_config: Optional[Dict[str, Any]] = None,
        ds_min_confidence: float = 0.3,
        ds_max_dist: float = 0.2,
        ds_max_iou_distance: float = 0.7,
        ds_max_age: int = 70,
        ds_n_init: int = 3,
        ds_nn_budget: int = 100,
        use_cuda: bool = True,
        # 输入预处理
        norm_mode: str = "db",   # 'db' | 'minmax' | 'none'
        # 控制
        deploy: bool = True,
        verbose: bool = False
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.norm_mode = norm_mode
        self.verbose = verbose

        # ----------------- 构建检测器 -----------------
        self.detector = YOLORTv1(
            cfg=cfg_model,
            device=self.device,
            img_size=self.img_size,
            num_classes=num_classes,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            trainable=False,
            deploy=deploy
        ).to(self.device).eval()

        if weight_path is not None and os.path.isfile(weight_path):
            ckpt = torch.load(weight_path, map_location=self.device)
            state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.detector.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print(f"[Detector] Loaded weights: {weight_path}")
                if missing: print(f"  Missing keys: {missing}")
                if unexpected: print(f"  Unexpected keys: {unexpected}")
        else:
            if self.verbose:
                print("[Detector] No weight loaded (using random init or pretrained backbone only).")

        # ----------------- 构建 DeepSORT 跟踪器 -----------------
        if deepsort_model_path is None:
            default_reid = os.path.join("deep_sort", "deep", "checkpoint", "ckpt.t7")
            deepsort_model_path = default_reid if os.path.isfile(default_reid) else None
        self.deepsort = DeepSort(
            model_path=deepsort_model_path,
            model_config=deepsort_model_config,
            max_dist=ds_max_dist,
            min_confidence=ds_min_confidence,
            nms_max_overlap=1.0,
            max_iou_distance=ds_max_iou_distance,
            max_age=ds_max_age,
            n_init=ds_n_init,
            nn_budget=ds_nn_budget,
            use_cuda=(use_cuda and torch.cuda.is_available())
        )

        # 状态缓存
        self.last_dets: Optional[np.ndarray] = None
        self.last_tracks: Optional[np.ndarray] = None

    # ----------------- 工具：输入预处理 -----------------
    def _normalize_input(self, img: np.ndarray) -> np.ndarray:
        """
        输入可以是 [H,W], [C,H,W], [H,W,C]。输出 float32 [C,H,W] 并 resize 到 self.img_size。
        """
        if img.ndim == 2:
            arr = img.astype(np.float32)
            if self.norm_mode == "db":
                eps = 1e-6
                db = 20.0 * np.log10(np.abs(arr) + eps)
                db_min, db_max = np.percentile(db, [1, 99])
                arr = np.clip((db - db_min) / (db_max - db_min + 1e-6), 0.0, 1.0)
            elif self.norm_mode == "minmax":
                vmin, vmax = np.percentile(arr, [1, 99])
                arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            elif self.norm_mode == "none":
                arr = arr
            else:
                # 默认全局 min-max
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            arr = arr[None, ...]  # [1,H,W]
        elif img.ndim == 3:
            if img.shape[0] in (1, 2, 3) and img.shape[0] < img.shape[1]:
                arr = img.astype(np.float32)
            else:
                arr = img.astype(np.float32).transpose(2, 0, 1)
            arr_min, arr_max = np.percentile(arr, [1, 99])
            arr = np.clip((arr - arr_min) / (arr_max - arr_min + 1e-6), 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported image ndim: {img.ndim}")

        _, H, W = arr.shape
        th, tw = self.img_size
        if (H, W) != (th, tw):
            # Resize 使用双线性
            tmp = arr.transpose(1, 2, 0)
            tmp = cv2.resize(tmp, (tw, th), interpolation=cv2.INTER_LINEAR)
            arr = tmp.transpose(2, 0, 1)
        return arr.astype(np.float32)

    @staticmethod
    def _xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = b
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([cx, cy, w, h], dtype=np.float32)

    # ----------------- NMS 实现 -----------------
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> List[int]:
        if boxes.numel() == 0:
            return []
        if tv_nms is not None:
            keep = tv_nms(boxes, scores, iou_thr)
            return keep.cpu().tolist()
        # 纯 PyTorch / NumPy fallback
        b = boxes.cpu().numpy()
        s = scores.cpu().numpy()
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_o = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
            union = area_i + area_o - inter + 1e-6
            iou = inter / union
            remain = np.where(iou <= iou_thr)[0]
            order = order[remain + 1]
        return keep

    # ----------------- 检测（单帧） -----------------
    @torch.no_grad()
    def detect(self, image: np.ndarray, return_tensor: bool = False) -> np.ndarray:
        """
        输入: numpy 图像 (支持灰度 / 多通道 / 复合并行处理后格式)
        输出: [N,6] -> [x1,y1,x2,y2,score,cls]
        """
        arr = self._normalize_input(image)
        tensor = torch.from_numpy(arr)[None, ...].to(self.device).float()

        outputs = self.detector(tensor)

        dets_list: List[List[float]] = []

        # deploy 模式：张量 [M,5]
        if isinstance(outputs, torch.Tensor):
            # outputs: [M,5] => [x1,y1,x2,y2,obj_score]
            if outputs.numel() == 0:
                self.last_dets = np.zeros((0, 6), dtype=np.float32)
                return self.last_dets
            boxes = outputs[:, :4]
            scores = outputs[:, 4]
            # 阈值
            keep = scores >= self.conf_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            labels = torch.zeros_like(scores, dtype=torch.int64)
            # NMS
            keep_idx = self._nms(boxes, scores, self.nms_thresh)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i].tolist()
                dets_list.append([x1, y1, x2, y2, float(scores[i]), int(labels[i])])
        # 非 deploy： (bboxes, scores, labels) -> numpy
        elif isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            bboxes, scores, labels = outputs
            if bboxes is None or len(bboxes) == 0:
                self.last_dets = np.zeros((0, 6), dtype=np.float32)
                return self.last_dets
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = map(float, bboxes[i])
                dets_list.append([x1, y1, x2, y2, float(scores[i]), int(labels[i])])
        else:
            raise RuntimeError("Unexpected detector output format.")

        if len(dets_list) == 0:
            self.last_dets = np.zeros((0, 6), dtype=np.float32)
        else:
            dets = np.array(dets_list, dtype=np.float32)
            # 过滤非法框
            finite = np.isfinite(dets).all(axis=1)
            good = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
            dets = dets[finite & good]
            self.last_dets = dets

        if return_tensor:
            return torch.from_numpy(self.last_dets)
        return self.last_dets

    # 别名
    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.detect(image)

    # ----------------- 跟踪（单帧） -----------------
    @torch.no_grad()
    def track(self, image: np.ndarray) -> np.ndarray:
        """
        输出: [M,6] -> [x1,y1,x2,y2,cls,track_id]  (int32)
        """
        dets = self.detect(image)  # [N,6]
        if dets.shape[0] == 0:
            self.last_tracks = np.zeros((0, 6), dtype=np.int32)
            return self.last_tracks

        b_xyxy = dets[:, :4]
        scores = dets[:, 4]
        classes = dets[:, 5].astype(np.int32)

        b_xywh = np.stack([self._xyxy_to_xywh(b) for b in b_xyxy], axis=0)

        # 适配 ReID 输入：保证 3 通道
        ori_img = image
        if ori_img.ndim == 2:
            base = np.clip(ori_img, 0, None)
            base = (base - base.min()) / (base.max() - base.min() + 1e-6)
            base_uint8 = (base * 255.0).astype(np.uint8)
            ori_img = np.stack([base_uint8, base_uint8, base_uint8], axis=-1)
        elif ori_img.ndim == 3:
            if ori_img.shape[0] in (1, 2, 3) and ori_img.shape[0] < ori_img.shape[1]:
                ori_img = ori_img.transpose(1, 2, 0)
            if ori_img.shape[2] == 1:
                ori_img = np.repeat(ori_img, 3, axis=2)
            elif ori_img.shape[2] == 2:
                third = np.mean(ori_img, axis=2, keepdims=True)
                ori_img = np.concatenate([ori_img, third], axis=2)
            # 归一化到 0-255
            if ori_img.dtype != np.uint8:
                mm = (ori_img - ori_img.min()) / (ori_img.max() - ori_img.min() + 1e-6)
                ori_img = (mm * 255.0).clip(0, 255).astype(np.uint8)
        else:
            raise ValueError("Unsupported image ndim for tracking input.")

        outputs, _ = self.deepsort.update(b_xywh, scores, classes, ori_img, masks=None)
        if outputs is None or (isinstance(outputs, np.ndarray) and outputs.size == 0):
            self.last_tracks = np.zeros((0, 6), dtype=np.int32)
        else:
            self.last_tracks = outputs
        return self.last_tracks

    # ----------------- 批量检测 -----------------
    def batch_detect(self, images: Iterable[np.ndarray]) -> List[np.ndarray]:
        results = []
        for img in images:
            results.append(self.detect(img))
        return results

    # ----------------- 序列跟踪 -----------------
    def track_sequence(self, images: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        for img in images:
            yield self.track(img)

    # ----------------- 重置跟踪器 -----------------
    def reset_tracker(self) -> None:
        """
        清空 DeepSORT 内部状态 (用于新序列或评估开始)。
        """
        self.deepsort.tracker.tracks = []
        self.deepsort.tracker._next_id = 1
        self.last_tracks = None

    # ----------------- 调试辅助 -----------------
    def summary(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "img_size": self.img_size,
            "conf_thresh": self.conf_thresh,
            "nms_thresh": self.nms_thresh,
            "norm_mode": self.norm_mode,
            "deploy": getattr(self.detector, "deploy", False),
            "last_dets": None if self.last_dets is None else self.last_dets.shape,
            "last_tracks": None if self.last_tracks is None else self.last_tracks.shape
        }


if __name__ == "__main__":
    # 简易自测 (需准备一张矩阵或图像)
    dummy = np.random.randn(600, 800).astype(np.float32)
    # 伪配置 (需替换为真实 cfg.model.to_dict())
    cfg_model = {
        "in_channels": 1,
        "pretrained": False,
        "expand_ratio": 0.5,
        "pooling_size": 5,
        "neck_act": "lrelu",
        "neck_norm": "BN",
        "num_cls_head": 2,
        "num_reg_head": 2,
        "head_act": "lrelu",
        "head_norm": "BN",
        "head_depthwise": False,
        "loss_obj_weight": 1.0,
        "loss_box_weight": 5.0
    }
    tracker = RadarTargetTracker(cfg_model, num_classes=1, deploy=True, verbose=True)
    dets = tracker.detect(dummy)
    print("Detections:", dets.shape)
    tracks = tracker.track(dummy)
    print("Tracks:", tracks.shape)
    print(tracker.summary())