"""
auto_encoder.py

使用 Counterfactual-LatentSpaceSharing-LFRecon 的预处理和数据加载方式，
训练一个卷积自编码器 AE_Vanilla。

- 输入来自 data_preprocess.get_ae_dataloader：
  images: (B, K, 1, H, W), 像素范围 [0, 1]
- 训练时映射到 [-1, 1] 喂给 AE（因为最后一层是 Tanh）
"""

import argparse
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_preprocess import get_ae_dataloader  # ⚠ 确保文件名是 data_preprocess.py


class AE_Vanilla(nn.Module):
    """
    卷积自编码器：
    - encoder: 多层 Conv2d + BN + LeakyReLU, stride=2 下采样
    - decoder: 对称 ConvTranspose2d 上采样
    - 输出: Tanh, 范围 [-1, 1]

    hidden_dims 决定每层通道数，最后一层的通道数就是 latent 的 channel 数。
    """

    def __init__(self, in_channels: int = 1, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            # 你可以按需要改成 [16, 32, 64, 16]，让最后通道数 = 16
            hidden_dims = [16, 32, 64, 128]

        out_channels = in_channels

        # ---------- Encoder ----------
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # ---------- Decoder ----------
        modules = []
        hidden_dims_rev = list(hidden_dims)
        hidden_dims_rev.reverse()  # 原地反转

        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_rev[i],
                        hidden_dims_rev[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims_rev[i + 1]),
                    nn.LeakyReLU(inplace=True),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_rev[-1],
                hidden_dims_rev[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims_rev[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                hidden_dims_rev[-1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_rec = self.decoder(z)
        x_rec = self.final_layer(x_rec)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(
    data_root: str,
    sequence_type: str = "all",                 # 兼容老接口
    sequence_types: Optional[List[str]] = None, # 新：自定义序列列表
    target_size: Tuple[int, int] = (256, 256),
    domain_mode: str = "both",                  # 低场 / 高场 / 两者一起
    batch_size: int = 4,
    num_workers: int = 4,
    num_epochs: int = 50,
    lr: float = 1e-4,
    hidden_dims: Optional[List[int]] = None,
    gpu_id: int = 0,
    out_dir: str = "ae_results",
    preload: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # ------- 设备设置：GPU 选择 -------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, using CPU.")

    # ------- DataLoader -------
    train_loader = get_ae_dataloader(
        data_root=data_root,
        sequence_type=sequence_type,
        sequence_types=sequence_types,
        target_size=target_size,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        preload=preload,
        domain_mode=domain_mode,
    )

    val_loader = get_ae_dataloader(
        data_root=data_root,
        sequence_type=sequence_type,
        sequence_types=sequence_types,
        target_size=target_size,
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        preload=preload,
        domain_mode=domain_mode,
        shuffle=False,
    )

    # ------- Model -------
    model = AE_Vanilla(in_channels=1, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 用 L1 重建损失（你也可以改成 MSE）
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ===== Train =====
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}] Train")

        for batch in pbar:
            images = batch["images"]  # (B, K, 1, H, W)
            B, K, C, H, W = images.shape

            # 展平 (B * K, 1, H, W)，把 low/high 一起当作样本
            images = images.view(B * K, C, H, W).to(device)  # in [0, 1]

            # 映射到 [-1, 1]，适配 Tanh 输出
            imgs_in = images * 2.0 - 1.0

            optimizer.zero_grad()
            recon = model(imgs_in)

            loss = criterion(recon, imgs_in)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(train_count, 1)
        writer.add_scalar("loss/train", avg_train_loss, epoch)

        # ===== Val =====
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch}/{num_epochs}] Val"):
                images = batch["images"]  # (B, K, 1, H, W)
                B, K, C, H, W = images.shape
                images = images.view(B * K, C, H, W).to(device)

                imgs_in = images * 2.0 - 1.0
                recon = model(imgs_in)

                loss = criterion(recon, imgs_in)
                val_loss_sum += loss.item() * images.size(0)
                val_count += images.size(0)

        avg_val_loss = val_loss_sum / max(val_count, 1)
        writer.add_scalar("loss/val", avg_val_loss, epoch)

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

        # 保存最好模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(ckpt_dir, "best_ae.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "hidden_dims": hidden_dims,
                },
                ckpt_path,
            )
            print(f"  >> Best model updated: {ckpt_path}")

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train AE_Vanilla on MRI dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/data54/wanghaobo/data/Third_full/",
    )
    # 兼容老接口：只用一个序列 or all
    parser.add_argument(
        "--sequence_type",
        type=str,
        default="all",
        choices=["fse", "irfse", "se", "all"],
        help="单序列 or all（若同时给了 --seq_list，则以 seq_list 为准）",
    )
    # 新：自定义多个序列
    parser.add_argument(
        "--seq_list",
        type=str,
        nargs="+",
        default=None,
        help="自定义序列列表，例如: --seq_list fse irfse",
    )
    parser.add_argument("--target_h", type=int, default=256)
    parser.add_argument("--target_w", type=int, default=256)
    parser.add_argument(
        "--domain_mode",
        type=str,
        default="both",
        choices=["low", "high", "both"],
        help="训练时用 0.5T / 1.5T / 两者都用",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="使用的 GPU 编号，如 0、1、2 ...",
    )
    parser.add_argument("--out_dir", type=str, default="ae_results")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Encoder channels list; 最后一个是 latent 通道数",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all data into memory to speed up training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_size = (args.target_h, args.target_w)

    # 如果给了 seq_list，就以 seq_list 为准
    seq_list = args.seq_list
    if seq_list is not None:
        print(f"Using custom sequence list: {seq_list}")
    else:
        print(f"Using sequence_type = {args.sequence_type} (no custom seq_list)")

    print("Training AE_Vanilla with config:")
    print(f"  data_root     = {args.data_root}")
    print(f"  target_size   = {target_size}")
    print(f"  sequence_type = {args.sequence_type}")
    print(f"  seq_list      = {seq_list}")
    print(f"  domain_mode   = {args.domain_mode}")
    print(f"  batch_size    = {args.batch_size}")
    print(f"  num_epochs    = {args.num_epochs}")
    print(f"  lr            = {args.lr}")
    print(f"  hidden_dims   = {args.hidden_dims}")
    print(f"  gpu_id        = {args.gpu_id}")
    print(f"  preload       = {args.preload}")

    train_autoencoder(
        data_root=args.data_root,
        sequence_type=args.sequence_type,
        sequence_types=seq_list,
        target_size=target_size,
        domain_mode=args.domain_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        gpu_id=args.gpu_id,
        out_dir=args.out_dir,
        preload=args.preload,
    )
