import torch
from dataset.radar_dataset import RadarWindowDataset
from torch.utils.data import DataLoader

class RadarTargetTracker:
    def __init__(self, cfg, dataset, detector, tracker):
        self.cfg = cfg
        self.dataset = dataset
        self.detector = detector
        self.tracker = tracker
        self.loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=RadarWindowDataset.collate_fn,
                                 pin_memory=True)
    def run(self):
        device = self.cfg.device
        self.detector.to(device)
        self.detector.eval()

        for batch in self.loader:
            image = batch["images"].to(device, non_blocking=True).float()
            targets=batch["targets"]
            metas = batch.get("batch_meta", None)

            # 目标检测
            with torch.no_grad():
                outputs = self.detector(image)

            # 提取检测结果
            pred_bboxes, pred_scores = outputs[0], outputs[1]

            # 可视化单帧检测结果
            from utils.visualize import visualize_single_frame_dets_and_gt
            # visualize_single_frame_dets_and_gt(image, pred_bboxes, pred_scores, targets[0])
            visualize_single_frame_dets_and_gt(image, pred_bboxes, pred_scores, batch['raw_boxes'],batch["targets"][0])


            # 准备 DeepSort 输入格式
            detections = []
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                detections.append([x1, y1, width, height, score, label])

            # 更新跟踪器
            tracks = self.tracker.update(detections)

            # 输出跟踪结果
            for track in tracks:
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                print(f"Track ID: {track_id}, BBox: {ltrb}")
