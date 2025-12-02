from data_augment import build_radar_augmentation
from radar_dataset import RadarWindowDataset

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.visualize import (visualize_batch_with_full, compute_and_visualize_stats)


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置

if __name__ == "__main__":
    
    augment = build_radar_augmentation(
    norm_mode="standard",
    vertical_flip_prob=1,
    clutter_prob=1,
    snr_range=(20.0, 21.0),
    background_only=False
    )
    train_dataset = RadarWindowDataset(
        mat_dir=".\dataset\data",
        csv_path=".\dataset\data\data_mat.csv",
        window_size=(640, 640),
        stride=(640, 640),
        complex_mode="abs",
        class_mapping={"mt": 0, "wm": 1, "uk":2},
        cache_mat_files=8,
        transform=None,
        subset='test',
        azimuth_split_ratio=0.7,
        full_frame=True
    )  
    # from utils.visualize import analyze_and_visualize_wm_diff
    # stats_wm = analyze_and_visualize_wm_diff(
    #     train_dataset,
    #     tolerance_pos=5.0,
    #     tolerance_size_ratio=0.5,
    #     reference_group=1,
    #     topn_frames=10,
    #     show=True
    # )

    # 统计与可视化
    # stats = compute_and_visualize_stats(train_dataset)
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=RadarWindowDataset.collate_fn)

    for batch in loader:
            visualize_batch_with_full(train_dataset, batch, complex_mode="abs", max_per_batch=8)

