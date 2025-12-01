from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Tuple, Dict, Optional, Any
import json
import os


# ---------------------------- 数据配置 ----------------------------
# ---------------------------- 增强配置 ----------------------------
@dataclass
class AugmentConfig:
    norm_mode: str = "standard"
    vertical_flip_prob: float = 0.5
    clutter_prob: float = 0.3
    snr_range: Tuple[float, float] = (0.0, 20.0)
    background_only: bool = False
    mosaic: Optional[float] = None
    mixup: Optional[float] = None

@dataclass
class DataConfig:
    mat_dir: str = "./dataset/data"
    csv_path: str = "./dataset/data/data_mat.csv"
    window_size_train: Tuple[int, int] = (640, 640)
    stride_train: Tuple[int, int] = (160, 160)
    window_size_test: Tuple[int, int] = (640, 640)
    stride_test: Tuple[int, int] = (640, 640)
    complex_mode: str = "abs"
    class_mapping: Dict[str, int] = field(default_factory=lambda: {"mt": 0, "wm": 1, "uk": 2})
    cache_mat_train: int = 64
    cache_mat_test: int = 4
    azimuth_split_ratio: float = 0.7

    @property
    def num_classes(self) -> int:
        return len(self.class_mapping)

# ---------------------------- 训练配置 ----------------------------
@dataclass
class TrainConfig:
    max_epoch: int = 20
    warmup_epoch: int = 3
    eval_interval: int = 1
    no_aug_epoch: int = 20
    grad_accumulate: int = 1          # 若使用自动计算，会在代码里覆盖
    batch_size: int = 64
    shuffle: bool = False
    img_size: int = 640
    fp16: bool = False
    clip_grad: float = 10.0
    persistent_workers: bool = True
    num_workers_train: int = 8
    num_workers_test: int = 1


# ---------------------------- 评估与保存配置 ----------------------------
@dataclass
class EvalConfig:
    conf_thresh: float = 0.8
    nms_thresh: float = 0.6
    topk: int = 1000
    iou_thresh: float = 0.5
    save_folder: str = "weights/"
    resume: Optional[str] = None
    pretrained_weights: Optional[str] = None

# ---------------------------- 模型配置 ----------------------------
@dataclass
class ModelConfig:
    backbone: str = "resnet18"  # 目前硬编码的结构，保留字段以便未来扩展
    pretrained: bool = True
    in_channels: int = 1
    neck: str = "sppf"
    expand_ratio: float = 0.5
    pooling_size: int = 5
    neck_act: str = "lrelu"
    neck_norm: str = "BN"
    num_cls_head: int = 2
    num_reg_head: int = 2
    head_act: str = "silu"
    head_norm: str = "BN"
    head_depthwise: bool = False
    loss_obj_weight: float = 1.0
    loss_cls_weight: float = 0.0  # 当前未使用分类损失
    loss_box_weight: float = 5.0
    loss_obj_empty_factor: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------- 优化器配置 ----------------------------
@dataclass
class OptimConfig:
    optimizer: str = "sgd"       # sgd | adamw | adam
    momentum: float = 0.937
    weight_decay: float = 5e-4
    lr0: float = 0.01


# ---------------------------- 学习率调度 ----------------------------
@dataclass
class SchedulerConfig:
    scheduler: str = "linear"    # linear | cosine
    lrf: float = 0.01            # final_lr = lr0 * lrf
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1




# ---------------------------- 总实验配置 ----------------------------
@dataclass
class ExperimentConfig:
    device: str = "cuda"
    distributed: bool = False
    dist_url: str = "env://"
    world_size: int = 1
    seed: int = 42
    vis_tgt: bool = False
    vis_aux_loss: bool = False
    tfboard: bool = False
    vis_pred_full: bool = True

    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def summary(self) -> str:
        lines = ["===== Experiment Configuration ====="]
        def block(name: str, obj: Any):
            lines.append(f"[{name}]")
            if is_dataclass(obj):
                items = asdict(obj).items()
            elif isinstance(obj, dict):
                items = obj.items()
            else:
                items = [("value", obj)]
            for k, v in items:
                lines.append(f"  {k}: {v}")
        # 顶层实验参数
        exp_keys = ["device","distributed","dist_url","world_size","seed",
                    "vis_tgt","vis_aux_loss","tfboard","vis_pred_full"]
        block("experiment", {k: getattr(self, k) for k in exp_keys})
        # 嵌套配置
        block("data", self.data)
        block("augment", self.augment)
        block("model", self.model)
        block("optim", self.optim)
        block("scheduler", self.scheduler)
        block("train", self.train)
        block("eval", self.eval)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Config] Saved configuration to {path}")


def load_config(path: Optional[str]) -> ExperimentConfig:
    """Load configuration from a JSON file; if path is None, return default config."""
    if path is None:
        return ExperimentConfig()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Reconstruct nested dataclasses
    cfg = ExperimentConfig(
        device=data.get('device', 'cuda'),
        distributed=data.get('distributed', False),
        dist_url=data.get('dist_url', 'env://'),
        world_size=data.get('world_size', 1),
        seed=data.get('seed', 42),
        vis_tgt=data.get('vis_tgt', False),
        vis_aux_loss=data.get('vis_aux_loss', False),
        tfboard=data.get('tfboard', False),
        vis_pred_full=data.get('vis_pred_full', False),
        data=DataConfig(**data.get('data', {})),
        augment=AugmentConfig(**data.get('augment', {})),
        model=ModelConfig(**data.get('model', {})),
        optim=OptimConfig(**data.get('optim', {})),
        scheduler=SchedulerConfig(**data.get('scheduler', {})),
        train=TrainConfig(**data.get('train', {})),
        eval=EvalConfig(**data.get('eval', {})),
    )
    return cfg


__all__ = [
    'DataConfig', 'AugmentConfig', 'ModelConfig', 'OptimConfig', 'SchedulerConfig',
    'TrainConfig', 'EvalConfig', 'ExperimentConfig', 'load_config'
]
