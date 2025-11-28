import os
import numpy as np
import torch
from utils.box_ops import box_iou, box_cxcywh_to_xyxy
from dataset.radar_dataset import RadarWindowDataset
from config import ExperimentConfig
from utils.visualize import visualize_full_predictions

class Evaluator:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.iou_thresh = cfg.eval.iou_thresh

    @torch.no_grad()
    def evaluate(self, model, test_loader, device, epoch: int, best_map: float):
        # 切到评估模式与 GPU
        model.eval()
        # 暂时禁用训练前向（使用 inference 分支）
        prev_trainable = getattr(model, 'trainable', True)
        if hasattr(model, 'trainable'):
            model.trainable = False

        # 统计量初始化
        all_scores = []     # 每个预测的分数
        all_matches = []    # 每个预测是否为TP（1）或FP（0）
        prediction_records = []
        total_gt = 0        # 全部GT框数量
        total_tp = 0        # 全部TP数量（用于PD）
        total_fp = 0        # 全部FP数量（用于PFA）
        total_non_target_cells = 0  # 非目标方位-距离单元作为PFA分母

        for batch in test_loader:
            # images: [B,C,H,W], targets: List[Dict], batch_size=1
            images = batch["images"].to(device, non_blocking=True).float()
            H, W = images.shape[-2], images.shape[-1]
            metas = batch.get("batch_meta", None)
            # GT 框（归一化 cx,cy,w,h）转像素 xyxy
            targets = batch["targets"]
            tgt = targets[0]
            gt_cxcywh = tgt["boxes"]  # [N,4], 归一化
            gt_labels = tgt["labels"] # [N]
            if gt_cxcywh.numel() > 0:
                # 反归一化到像素
                gt_px = gt_cxcywh.clone()
                gt_px[:, 0] = gt_px[:, 0] * W  # cx
                gt_px[:, 1] = gt_px[:, 1] * H  # cy
                gt_px[:, 2] = gt_px[:, 2] * W  # w
                gt_px[:, 3] = gt_px[:, 3] * H  # h
                gt_xyxy = box_cxcywh_to_xyxy(gt_px)  # [N,4]
                # 为 PFA 分母统计目标单元掩膜
                mask = torch.zeros((H, W), dtype=torch.bool)
                for b in gt_xyxy:
                    x1 = int(torch.clamp(b[0], 0, W).item())
                    y1 = int(torch.clamp(b[1], 0, H).item())
                    x2 = int(torch.clamp(b[2], 0, W).item())
                    y2 = int(torch.clamp(b[3], 0, H).item())
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = True
                non_target_cells = int(H * W - mask.sum().item())
                total_non_target_cells += non_target_cells
            else:
                gt_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                total_non_target_cells += H * W

            total_gt += gt_xyxy.shape[0]

            # 模型推理（单类：仅用 obj 分数）
            preds = model(images)
            # 兼容返回类型：期望 (bboxes, scores, labels)
            if isinstance(preds, (tuple, list)) and len(preds) >= 2:
                pred_bboxes, pred_scores = preds[0], preds[1]
                # 可能是 numpy
                if isinstance(pred_bboxes, np.ndarray):
                    pred_bboxes = torch.from_numpy(pred_bboxes)
                if isinstance(pred_scores, np.ndarray):
                    pred_scores = torch.from_numpy(pred_scores)
            else:
                # 若模型未返回后处理结果，尝试空结果
                pred_bboxes = torch.zeros((0, 4), dtype=torch.float32)
                pred_scores = torch.zeros((0,), dtype=torch.float32)

            # 清理非法框
            if pred_bboxes.numel() > 0:
                valid = torch.isfinite(pred_bboxes).all(dim=1)
                proper = (pred_bboxes[:, 2] > pred_bboxes[:, 0]) & (pred_bboxes[:, 3] > pred_bboxes[:, 1])
                keep = valid & proper
                pred_bboxes = pred_bboxes[keep]
                pred_scores = pred_scores[keep]

            # 按分数排序
            if pred_scores.numel() > 0:
                order = torch.argsort(pred_scores, descending=True)
                pred_bboxes = pred_bboxes[order]
                pred_scores = pred_scores[order]

            # 匹配：贪心按预测遍历，与尚未匹配的GT中 IoU 最大的那个进行配对
            matched_gt = torch.zeros((gt_xyxy.shape[0],), dtype=torch.bool)
            for i in range(pred_bboxes.shape[0]):
                pb = pred_bboxes[i].unsqueeze(0)  # [1,4]
                score = float(pred_scores[i].item())
                if gt_xyxy.shape[0] > 0:
                    ious, _ = box_iou(pb, gt_xyxy)   # [1,N]
                    ious = ious.squeeze(0)           # [N]
                    # 将已匹配的GT屏蔽
                    ious[matched_gt] = -1.0
                    max_iou, max_j = (float(ious.max().item()), int(torch.argmax(ious).item()))
                    if max_iou >= self.iou_thresh:
                        # 计为TP，标记GT已匹配
                        matched_gt[max_j] = True
                        all_scores.append(score)
                        all_matches.append(1)
                        total_tp += 1
                    else:
                        # FP
                        all_scores.append(score)
                        all_matches.append(0)
                        total_fp += 1
                else:
                    # 无GT，全部FP
                    all_scores.append(score)
                    all_matches.append(0)
                    total_fp += 1

            # 注意：FN 可由 total_gt - 累计TP 推导，无需逐图统计
            if metas is not None:
                # 仅支持 batch_size=1 情况（当前数据集似乎如此）
                file_name = metas[0]["file"]
                y0, x0 = metas[0]["global_origin"]
                # 保存局部预测框与 GT（均为窗口内坐标）
                prediction_records.append({
                    "file": file_name,
                    "origin": (y0, x0),
                    "pred_boxes": pred_bboxes.cpu(),
                    "pred_scores": pred_scores.cpu(),
                    "gt_boxes": gt_xyxy.cpu(),  # 已是像素 xyxy（窗口内）
                })      

        # 计算 AP（单类）
        if len(all_scores) == 0:
            ap = 0.0
            pd = 0.0 if total_gt == 0 else 0.0
            pfa = 0.0
        else:
            scores = np.array(all_scores, dtype=np.float32)
            matches = np.array(all_matches, dtype=np.int32)

            # 按分数降序
            order = np.argsort(-scores)
            matches = matches[order]

            tp_cum = np.cumsum(matches)
            fp_cum = np.cumsum(1 - matches)

            # 召回与精度
            recall = tp_cum / max(total_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

            # 插值精度包络（从后向前取最大）
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])

            # 计算AP为P-R曲线面积
            # 仅在召回变化处累加
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

            # 检测概率（PD）
            pd = float(total_tp / max(total_gt, 1))

            # 虚警概率（PFA）：FP 个数 / 非目标方位-距离单元总数
            pfa = float(total_fp / max(total_non_target_cells, 1))

        # 保存最好权重
        if ap > best_map:
            os.makedirs(self.cfg.eval.save_folder, exist_ok=True)
            save_path = os.path.join(self.cfg.eval.save_folder, f"best_epoch_{epoch+1}_map_{ap:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            best_map = ap
            print(f"[Eval] New best mAP={ap:.4f}, PD={pd:.4f}, PFA={pfa:.6f} -> saved: {save_path}")
        else:
            print(f"[Eval] mAP={ap:.4f}, PD={pd:.4f}, PFA={pfa:.6f}")

        # 复原训练标记
        if hasattr(model, 'trainable'):
            model.trainable = prev_trainable
        model.train()
        if self.cfg.vis_pred_full and prediction_records:
            try:
                dataset = test_loader.dataset
                print("[Eval] 可视化整幅检测结果 (已缓存预测)")
                visualize_full_predictions(
                    dataset=dataset,
                    records=prediction_records,
                    conf_thresh=self.cfg.eval.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    max_files=10
                )
            except Exception as e:
                print(f"[Eval] 可视化失败: {e}")
        return best_map