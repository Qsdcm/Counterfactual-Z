"""
data_preprocess.py

数据预处理 & DataLoader 封装，用于训练自编码器。
复用原项目 dataset.py 里的 MRIPairedDataset / CombinedMRIDataset 处理流程：
- 从配对 k-space .mat 读取
- coil RSS 合成
- 归一化到 [0, 1]
- center crop 到指定大小
"""

import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset_utils import MRIPairedDataset, CombinedMRIDataset  # 来自原仓库


class AEImageDataset(Dataset):
    """
    自编码器用的数据集封装。

    内部调用：
      - 如果 sequence_types 不为 None -> CombinedMRIDataset(sequence_types=...)
      - 否则：
          * sequence_type='all' -> CombinedMRIDataset(默认 fse+irfse+se)
          * 其它 ('fse'/'irfse'/'se') -> MRIPairedDataset

    再对输出做一个小封装：
      - domain_mode='low'  -> 只用 0.5T (img_low)
      - domain_mode='high' -> 只用 1.5T (img_high)
      - domain_mode='both' -> 同时用 0.5T 和 1.5T

    __getitem__ 返回:
      images : (K, 1, H, W)  # K=1 或 2，H/W 固定为 target_size
      domains: (K,)          # 0.0 / 1.0
      file_idx, slice_idx:   原始索引，方便 debug
    """

    def __init__(
        self,
        data_root: str,
        sequence_type: str = "all",              # 兼容老接口
        sequence_types: Optional[List[str]] = None,  # 新：自定义序列列表
        target_size: Tuple[int, int] = (256, 256),
        split: str = "train",
        preload: bool = False,
        domain_mode: str = "low",
    ):
        assert domain_mode in ["low", "high", "both"], \
            f"domain_mode must be 'low', 'high' or 'both', got {domain_mode}"

        self.target_size = target_size
        self.domain_mode = domain_mode

        # ---------- 选择底层 dataset ----------
        if sequence_types is not None:
            # 使用自定义序列列表，例如 ['fse', 'irfse']
            print(f"[{split.upper()}] Using custom sequence list: {sequence_types}")
            self.base_dataset = CombinedMRIDataset(
                data_root=data_root,
                sequence_types=sequence_types,
                target_size=target_size,
                split=split,
                preload=preload,
            )
        else:
            if sequence_type == "all":
                # 默认：三个序列全用
                print(f"[{split.upper()}] Using all sequences: fse + irfse + se")
                self.base_dataset = CombinedMRIDataset(
                    data_root=data_root,
                    target_size=target_size,
                    split=split,
                    preload=preload,
                )
            else:
                # 单独某一序列：fse / irfse / se
                print(f"[{split.upper()}] Using single sequence: {sequence_type}")
                self.base_dataset = MRIPairedDataset(
                    data_root=data_root,
                    sequence_type=sequence_type,
                    target_size=target_size,
                    split=split,
                    preload=preload,
                    return_kspace=False,  # 自编码器只用图像
                )

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _crop_or_pad_to_target(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        对单个 CHW tensor 做中心裁剪 / padding 到统一大小 (target_h, target_w)
        img: (1, H, W)
        """
        _, H, W = img.shape

        # 1) 如果比目标大：中心裁剪
        if H > target_h:
            top = (H - target_h) // 2
            img = img[:, top:top + target_h, :]
            H = target_h
        if W > target_w:
            left = (W - target_w) // 2
            img = img[:, :, left:left + target_w]
            W = target_w

        # 2) 如果比目标小：中心 padding
        pad_h = max(target_h - H, 0)
        pad_w = max(target_w - W, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # F.pad(pad) 的顺序是 (left, right, top, bottom)
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

        return img

    def __getitem__(self, idx):
        base = self.base_dataset[idx]

        img_low = base["img_low"]    # (1, H_l, W_l), float32, [0, 1]
        img_high = base["img_high"]  # (1, H_h, W_h), float32, [0, 1]
        lab_low = base["label_low"]  # tensor([0.])
        lab_high = base["label_high"]  # tensor([1.])

        target_h, target_w = self.target_size

        # 额外一步：不管原始多大多小，都强行变成统一的 (target_h, target_w)
        img_low = self._crop_or_pad_to_target(img_low, target_h, target_w)
        img_high = self._crop_or_pad_to_target(img_high, target_h, target_w)

        if self.domain_mode == "low":
            images = img_low.unsqueeze(0)   # (1, 1, H, W)
            domains = lab_low.view(1)       # (1,)
        elif self.domain_mode == "high":
            images = img_high.unsqueeze(0)  # (1, 1, H, W)
            domains = lab_high.view(1)      # (1,)
        else:  # both
            images = torch.stack([img_low, img_high], dim=0)   # (2, 1, H, W)
            domains = torch.stack([lab_low, lab_high], dim=0).view(2)  # (2,)

        return {
            "images": images,                     # (K, 1, target_h, target_w)
            "domains": domains,                   # (K,)
            "file_idx": base["file_idx"],
            "slice_idx": base["slice_idx"],
        }


def get_ae_dataloader(
    data_root: str,
    sequence_type: str = "all",                  # 兼容老接口
    sequence_types: Optional[List[str]] = None,  # 新：自定义序列列表
    target_size: Tuple[int, int] = (256, 256),
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    preload: bool = False,
    domain_mode: str = "both",
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    为自编码器创建 DataLoader。

    Args:
        data_root: 数据根目录
        sequence_type: 'fse' | 'irfse' | 'se' | 'all'（与 sequence_types 二选一）
        sequence_types: 自定义序列列表，例如 ['fse', 'irfse']
        target_size: 输出图像大小 (H, W)
        split: 'train' 或 'test'
        batch_size: batch 大小
        num_workers: DataLoader worker 数
        preload: 是否预加载到内存
        domain_mode: 'low' | 'high' | 'both'
        shuffle: 是否打乱，默认 train=True, test=False
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = AEImageDataset(
        data_root=data_root,
        sequence_type=sequence_type,
        sequence_types=sequence_types,
        target_size=target_size,
        split=split,
        preload=preload,
        domain_mode=domain_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader


if __name__ == "__main__":
    # 简单自测：打印一个 batch 的形状
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--sequence_type", type=str, default="all",
                        choices=["fse", "irfse", "se", "all"])
    parser.add_argument("--seq_list", type=str, nargs="+", default=None,
                        help="自定义序列列表，例如: --seq_list fse irfse")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--domain_mode", type=str, default="both",
                        choices=["low", "high", "both"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    dl = get_ae_dataloader(
        data_root=args.data_root,
        sequence_type=args.sequence_type,
        sequence_types=args.seq_list,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        domain_mode=args.domain_mode,
    )

    batch = next(iter(dl))
    images = batch["images"]  # (B, K, 1, H, W)
    domains = batch["domains"]

    print(f"images shape: {images.shape}")   # e.g. (2, 2, 1, 256, 256)
    print(f"domains shape: {domains.shape}") # e.g. (2, 2)
