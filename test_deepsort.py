import argparse
from config import  load_config
import torch

from dataset.radar_dataset import RadarWindowDataset
from models.yoloRTv1.yolort import YOLORTv1
from deep_sort.deep_sort import DeepSort
from trackers.radar_target_tracker import RadarTargetTracker
from dataset.data_augment.radar_augment import PadToStride
from dataset.data_augment.radar_augment import AmplitudeNormalize
from dataset.data_augment.radar_augment import Compose
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Radar Training')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file.')
    parser.add_argument('--device', type=str, default=None, help='Override device, e.g. cpu/cuda.')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size.')
    parser.add_argument('--epochs', type=int, default=None, help='Override max epochs.')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming.')
    parser.add_argument('--save_folder', type=str, default=None, help='Override save folder.')
    parser.add_argument('--conf_thresh', type=float, default=None, help='Override confidence threshold.')
    parser.add_argument('--nms_thresh', type=float, default=None, help='Override NMS threshold.')
    parser.add_argument('--iou_thresh', type=float, default=None, help='Override IoU threshold for eval.')
    return parser.parse_args()


    
    
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg.apply_seed(deterministic=cfg.deterministic)
    # 覆盖简单参数
    if args.device: cfg.device = args.device
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.epochs: cfg.train.max_epoch = args.epochs
    if args.fp16: cfg.train.fp16 = True
    if args.resume: cfg.eval.resume = args.resume
    if args.save_folder: cfg.eval.save_folder = args.save_folder
    if args.conf_thresh: cfg.eval.conf_thresh = args.conf_thresh
    if args.nms_thresh: cfg.eval.nms_thresh = args.nms_thresh
    if args.iou_thresh: cfg.eval.iou_thresh = args.iou_thresh

    deepsort_dataset = RadarWindowDataset(
        mat_dir=".\dataset\data",
        csv_path=".\dataset\data\data_mat.csv",
        complex_mode="abs",
        class_mapping={"mt": 0, "wm": 1, "uk":2},
        cache_mat_files=64,
        transform=Compose([AmplitudeNormalize(mode=cfg.augment.norm_mode), 
                           PadToStride(stride=16, pad_value=0.0)]),
        subset='test',
        azimuth_split_ratio=0.7,
        full_frame=True
    )
    detector=YOLORTv1(
            cfg=cfg.model.to_dict(),
            device=cfg.device,
            num_classes=cfg.data.num_classes,
            conf_thresh=cfg.eval.conf_thresh,
            nms_thresh=cfg.eval.nms_thresh)
    weights_path="./weights/best_epoch_35_map_0.4619.pth"
    state_dict = torch.load(weights_path, map_location=cfg.device)
    detector.load_state_dict(state_dict)

    radar_target_tracker = RadarTargetTracker(
        cfg=cfg,
        dataset=deepsort_dataset,
        detector=detector,
        tracker=DeepSort(
            model_path=cfg.deepsort.model_path,
            max_dist=cfg.deepsort.max_dist,
            max_iou_distance=cfg.deepsort.max_iou_distance,
            max_age=cfg.deepsort.max_age,
            n_init=cfg.deepsort.n_init,
            nn_budget=cfg.deepsort.nn_budget,
            use_cuda=(cfg.device != 'cpu')
        )
    )
    radar_target_tracker.run()