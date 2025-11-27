"""统一的训练入口，使用独立的 Trainer 与 Evaluator 类。"""

import argparse
import torch

from config import ExperimentConfig, load_config
from engine import Trainer, Evaluator


# ---------------------------- 命令行解析（仅覆盖关键参数） ----------------------------
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


# ---------------------------- 数据集与加载器构建 ----------------------------
# 数据集/模型等构建由 Trainer 封装


# DataLoader 构建由 Trainer 封装


# ---------------------------- 模型与损失构建 ----------------------------
# 模型与损失由 Trainer 封装


# ---------------------------- 评估函数（单类 AP 用对象ness） ----------------------------
# 评估逻辑由 Evaluator 封装


# ---------------------------- 单 epoch 训练 ----------------------------
# 单 epoch 训练由 Trainer 封装


# ---------------------------- 主函数流水线 ----------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)
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

    print(cfg.summary())
    trainer = Trainer(cfg)
    evaluator = Evaluator(cfg)
    best_map = trainer.run(evaluator=evaluator)
    print("Training finished. Best mAP(single-class): {:.4f}".format(best_map))


if __name__ == '__main__':
    main()

        
