import torch
import torch.nn.functional as F

from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import YoloMatcher


class Criterion(object):
    def __init__(self, cfg, device, num_classes=1):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg.get('loss_obj_weight', 1.0)
        # 分类权重不再使用
        self.loss_cls_weight = 0.0
        self.loss_box_weight = cfg.get('loss_box_weight', 5.0)

        self.matcher = YoloMatcher(num_classes=num_classes)

    def loss_objectness(self, pred_obj, gt_obj):
        return F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')
    
    def loss_bboxes(self, pred_box, gt_box):
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        return 1.0 - ious

    def __call__(self, outputs, targets, epoch=0):
        device = outputs['pred_obj'][0].device
        stride = outputs['stride']
        fmp_size = outputs['fmp_size']

        pred_obj = outputs['pred_obj'].view(-1)         # [BM,]
        pred_box = outputs['pred_box'].view(-1, 4)      # [BM,4]

        # 标签分配
        gt_objectness, gt_classes, gt_bboxes = self.matcher(
            fmp_size=fmp_size, stride=stride, targets=targets
        )
        gt_objectness = gt_objectness.view(-1).to(device).float()
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()

        pos_masks = (gt_objectness > 0)
        num_fgs = pos_masks.sum()

        # 空帧：仅训练 obj
        empty_frame = (num_fgs.item() == 0)
        if empty_frame:
            factor = self.cfg.get('loss_obj_empty_factor', 0.25)
            loss_obj_all = self.loss_objectness(pred_obj, gt_objectness)
            loss_obj = loss_obj_all.mean() * factor
            loss_box = pred_obj.new_tensor(0.0)
            losses = self.loss_obj_weight * loss_obj
            return {
                'loss_obj': loss_obj.detach(),
                'loss_box': loss_box.detach(),
                'losses': losses,
                'empty_frame': True
            }

        num_fgs = num_fgs.clamp(min=1).float()

        # objectness
        loss_obj = self.loss_objectness(pred_obj, gt_objectness).sum() / num_fgs

        # bbox regression
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_bboxes[pos_masks]
        loss_box = self.loss_bboxes(pred_box_pos, gt_bboxes_pos).sum() / num_fgs

        losses = (self.loss_obj_weight * loss_obj +
                  self.loss_box_weight * loss_box)

        return {
            'loss_obj': loss_obj,
            'loss_box': loss_box,
            'losses': losses,
            'empty_frame': False
        }

def build_criterion(cfg, device, num_classes):
    criterion = Criterion(cfg    = cfg,
                          device = device,
                          num_classes = num_classes
                          )

    return criterion

    
if __name__ == "__main__":
    pass
