import os
import numpy as np
import torch
from utils.box_ops import box_iou
from dataset.radar_dataset import RadarWindowDataset
from config import ExperimentConfig


class Evaluator:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.iou_thresh = cfg.eval.iou_thresh

    @torch.no_grad()
    def evaluate(self, model, test_loader, device, epoch: int, best_map: float):
        model.eval()
        prev_trainable = getattr(model, 'trainable', True)
        model.trainable = False

        stats = {'tp': [], 'fp': [], 'scores': [], 'n_gt': 0}

        for batch in test_loader:
            images = batch["images"].to(device).float()
            h, w = images.shape[-2:]
            targets = batch["targets"]

            gt_boxes_cxcywh = targets[0]["boxes"].cpu().numpy()
            gt_xyxy = []
            for (cx, cy, bw, bh) in gt_boxes_cxcywh:
                cx_pix = cx * w; cy_pix = cy * h
                bw_pix = bw * w; bh_pix = bh * h
                x1 = max(0.0, cx_pix - bw_pix * 0.5)
                y1 = max(0.0, cy_pix - bh_pix * 0.5)
                x2 = min(w, cx_pix + bw_pix * 0.5)
                y2 = min(h, cy_pix + bh_pix * 0.5)
                if x2 > x1 and y2 > y1:
                    gt_xyxy.append([x1, y1, x2, y2])
            gt_xyxy = np.array(gt_xyxy, dtype=np.float32)
            stats['n_gt'] += len(gt_xyxy)

            bboxes, scores, labels = model(images)
            if len(bboxes) == 0:
                continue
            order = np.argsort(-scores)
            bboxes = bboxes[order]
            scores = scores[order]
            matched = np.zeros(len(gt_xyxy), dtype=bool)
            for pb, ps in zip(bboxes, scores):
                stats['scores'].append(ps)
                if len(gt_xyxy) == 0:
                    stats['tp'].append(0); stats['fp'].append(1)
                    continue
                pb_t = torch.tensor(pb, dtype=torch.float32).unsqueeze(0)
                gt_t = torch.tensor(gt_xyxy, dtype=torch.float32)
                iou_mat, _ = box_iou(pb_t, gt_t)
                ious = iou_mat.squeeze(0).cpu().numpy()
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
                if best_iou >= self.iou_thresh and (not matched[best_idx]):
                    stats['tp'].append(1); stats['fp'].append(0)
                    matched[best_idx] = True
                else:
                    stats['tp'].append(0); stats['fp'].append(1)

        if stats['n_gt'] == 0 or len(stats['scores']) == 0:
            mAP = 0.0
        else:
            # 1) 将所有预测按置信度从高到低排序（模拟逐步放宽阈值的检索过程）
            scores_arr = np.array(stats['scores'])
            tp_arr = np.array(stats['tp'])
            fp_arr = np.array(stats['fp'])
            order = np.argsort(-scores_arr)
            tp_arr = tp_arr[order]
            fp_arr = fp_arr[order]

            # 2) 计算累积 TP/FP，得到每个阈值下的召回率与精确率曲线点
            #    tp_cum: 前 k 个预测有多少是真正例
            #    fp_cum: 前 k 个预测有多少是假正例
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            recalls = tp_cum / (stats['n_gt'] + 1e-9)              # 召回率 R = TP / GT
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)         # 精确率 P = TP / (TP + FP)

            # 3) 首尾补点，便于做“保序凸包”处理与区间积分
            mrec = np.concatenate(([0.], recalls, [1.]))
            mpre = np.concatenate(([0.], precisions, [0.]))

            # 4) 单调化精确率（保序）：确保 P(R) 随 R 单调不增，
            #    等价于对 PR 曲线做上包络，避免局部抖动影响 AP 估计
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])

            # 5) 插值积分：对 PR 曲线在召回的离散跳变处做面积累加
            #    AP = Σ (R_{i+1} - R_i) * P_{i+1}
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            mAP = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        print(f"[Eval][Epoch {epoch+1}] mAP@0.5(single-class): {mAP:.4f}")
        if mAP > best_map:
            os.makedirs(self.cfg.eval.save_folder, exist_ok=True)
            save_path = os.path.join(self.cfg.eval.save_folder, f"best_epoch_{epoch+1}_map_{mAP:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Eval] New best mAP improved from {best_map:.4f} to {mAP:.4f}, saved: {save_path}")
            best_map = mAP
        else:
            print(f"[Eval] mAP {mAP:.4f} did not improve best {best_map:.4f}")

        model.trainable = prev_trainable
        model.train()
        return best_map
