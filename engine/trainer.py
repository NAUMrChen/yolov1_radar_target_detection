import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.yoloRTv1.yolort import YOLORTv1
from models.yoloRTv1.loss import Criterion
from dataset.radar_dataset import RadarWindowDataset
from dataset.data_augment import build_radar_augmentation
from dataset.data_augment.radar_augment import AmplitudeNormalize
from utils.solver.optimizer import build_yolo_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler

from config import ExperimentConfig


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.fp16)
        self.model, self.criterion = self._build_model_and_loss()
        self.optimizer, self.start_epoch = build_yolo_optimizer(cfg.optim.__dict__, self.model, cfg.eval.resume)
        self.lr_scheduler, self.lf = build_lr_scheduler(cfg.scheduler.__dict__, self.optimizer, cfg.train.max_epoch)
        self.lr_scheduler.last_epoch = self.start_epoch - 1
        self.train_loader, self.test_loader = self._build_dataloaders()

    def _build_datasets(self):
        train_aug = build_radar_augmentation(
            norm_mode=self.cfg.augment.norm_mode,
            vertical_flip_prob=self.cfg.augment.vertical_flip_prob,
            clutter_prob=self.cfg.augment.clutter_prob,
            snr_range=self.cfg.augment.snr_range,
            background_only=self.cfg.augment.background_only,
        )
        train_dataset = RadarWindowDataset(
            mat_dir=self.cfg.data.mat_dir,
            csv_path=self.cfg.data.csv_path,
            window_size=self.cfg.data.window_size_train,
            stride=self.cfg.data.stride_train,
            complex_mode=self.cfg.data.complex_mode,
            class_mapping=self.cfg.data.class_mapping,
            cache_mat_files=self.cfg.data.cache_mat_train,
            transform=train_aug,
            subset='train',
            azimuth_split_ratio=self.cfg.data.azimuth_split_ratio,
        )
        test_dataset = RadarWindowDataset(
            mat_dir=self.cfg.data.mat_dir,
            csv_path=self.cfg.data.csv_path,
            window_size=self.cfg.data.window_size_test,
            stride=self.cfg.data.stride_test,
            complex_mode=self.cfg.data.complex_mode,
            class_mapping=self.cfg.data.class_mapping,
            cache_mat_files=self.cfg.data.cache_mat_test,
            transform=AmplitudeNormalize(mode=self.cfg.augment.norm_mode),
            subset='test',
            azimuth_split_ratio=self.cfg.data.azimuth_split_ratio,
        )
        return train_dataset, test_dataset

    def _build_dataloaders(self):
        train_dataset, test_dataset = self._build_datasets()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=train_dataset.__len__() if self.cfg.train.num_workers_train < 0 else self.cfg.train.num_workers_train,
            collate_fn=RadarWindowDataset.collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.train.persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.train.num_workers_test,
            collate_fn=RadarWindowDataset.collate_fn,
            pin_memory=True,
        )
        return train_loader, test_loader

    def _build_model_and_loss(self):
        model = YOLORTv1(
            cfg=self.cfg.model.to_dict(),
            device=self.device,
            img_size=self.cfg.train.img_size,
            num_classes=self.cfg.data.num_classes,
            conf_thresh=self.cfg.eval.conf_thresh,
            nms_thresh=self.cfg.eval.nms_thresh,
            trainable=True,
            deploy=False,
        ).to(self.device)
        criterion = Criterion(cfg=self.cfg.model.to_dict(), device=self.device, num_classes=self.cfg.data.num_classes)
        return model, criterion

    def train_one_epoch(self, epoch: int):
        epoch_size = len(self.train_loader)
        nw = epoch_size * self.cfg.train.warmup_epoch
        last_opt_step = getattr(self, 'last_opt_step', 0)
        t0 = time.time()
        for iter_i, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True).float()
            targets = batch["targets"]
            ni = iter_i + epoch * epoch_size
            accumulate = max(1, round(64 / self.cfg.train.batch_size))
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, 64 / self.cfg.train.batch_size]).round())
                for j, pg in enumerate(self.optimizer.param_groups):
                    pg['lr'] = np.interp(ni, xi, [self.cfg.scheduler.warmup_bias_lr if j == 0 else 0.0, pg['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in pg:
                        pg['momentum'] = np.interp(ni, xi, [self.cfg.scheduler.warmup_momentum, self.cfg.optim.momentum])
            with torch.cuda.amp.autocast(enabled=self.cfg.train.fp16):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs=outputs, targets=targets, epoch=epoch)
                losses = loss_dict['losses'] * images.shape[0]
            self.scaler.scale(losses).backward()
            if ni - last_opt_step >= accumulate:
                if self.cfg.train.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                last_opt_step = ni
                self.last_opt_step = last_opt_step
            if iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = self.optimizer.param_groups[2]['lr'] if len(self.optimizer.param_groups) > 2 else self.optimizer.param_groups[0]['lr']
                log = f"[Epoch: {epoch+1}/{self.cfg.train.max_epoch}][Iter: {iter_i}/{epoch_size}][lr: {cur_lr:.6f}]"
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        log += f"[{k}: {float(v):.2f}]"
                    else:
                        log += f"[{k}: {v}]"
                log += f"[time: {t1 - t0:.2f}][size: {self.cfg.train.img_size}]"
                print(log, flush=True)
                t0 = time.time()

    def run(self, evaluator=None):
        best_map = -1.0
        for epoch in range(self.start_epoch, self.cfg.train.max_epoch):
            self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            if (epoch % self.cfg.train.eval_interval) == 0 or epoch == self.cfg.train.max_epoch - 1:
                if evaluator is not None:
                    best_map = evaluator.evaluate(self.model, self.test_loader, self.device, epoch, best_map)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return best_map
