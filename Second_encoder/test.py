import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 路径配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
auto_encoder_path = os.path.join(parent_dir, 'auto_encoder')
sys.path.append(auto_encoder_path)

from data_preprocess import get_ae_dataloader
from model_paired import ContentEncoderVAE, Decoder

def psnr(x, y, max_val=2.0):
    mse = torch.mean((x - y) ** 2)
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()

def ssim_tensor(x, y):
    # 简易 SSIM 计算，建议实际科研中使用 pytorch-msssim 库
    # 这里复用之前给出的简单实现
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    mu_x = x.mean(); mu_y = y.mean()
    sigma_x = x.var(); sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (num / den).item()

def test(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.out_dir, "test_visualization")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载 Test 数据
    print("Loading TEST data...")
    test_loader = get_ae_dataloader(
        data_root=args.data_root,
        sequence_type=args.sequence_type,
        target_size=(256, 256),
        split="test",            # 使用 Test 文件夹
        batch_size=1,            # 逐张测试
        num_workers=4,
        domain_mode="both",
        shuffle=False
    )

    # 2. 加载模型
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    # 根据 checkpoint 反推网络通道数，避免加载时尺寸不匹配
    enc_dim = ckpt['E_A']['model.0.conv.weight'].shape[0]   # 例如 64
    dec_dim = ckpt['G_A']['model.0.model.0.model.0.conv.weight'].shape[0]  # 例如 256

    enc_params = {
        'n_downsample': 2,
        'n_res': 4,
        'input_dim': 1,
        'dim': enc_dim,
        'norm': 'in',
        'activ': 'relu',
        'pad_type': 'reflect'
    }
    dec_params = {
        'n_upsample': 2,
        'n_res': 4,
        'dim': dec_dim,
        'output_dim': 1,
        'res_norm': 'ln',
        'activ': 'relu',
        'pad_type': 'reflect'
    }
    
    E_A = ContentEncoderVAE(**enc_params).to(device)
    G_A = Decoder(**dec_params).to(device)
    E_B = ContentEncoderVAE(**enc_params).to(device)
    G_B = Decoder(**dec_params).to(device)
    
    E_A.load_state_dict(ckpt['E_A'])
    G_A.load_state_dict(ckpt['G_A'])
    E_B.load_state_dict(ckpt['E_B'])
    G_B.load_state_dict(ckpt['G_B'])
    
    E_A.eval(); G_A.eval()
    E_B.eval(); G_B.eval()

    # 3. 测试循环
    metrics = {'psnr_L': [], 'ssim_L': [], 'psnr_H': [], 'ssim_H': []}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            images = batch['images'] # (1, 2, 1, 256, 256)
            
            x_L = images[:, 0].to(device) * 2.0 - 1.0
            x_H = images[:, 1].to(device) * 2.0 - 1.0
            
            # Recon
            z_L, _, _ = E_A(x_L)
            rec_L = G_A(z_L)
            
            z_H, _, _ = E_B(x_H)
            rec_H = G_B(z_H)
            
            # Metrics
            metrics['psnr_L'].append(psnr(x_L, rec_L))
            metrics['ssim_L'].append(ssim_tensor(x_L, rec_L))
            metrics['psnr_H'].append(psnr(x_H, rec_H))
            metrics['ssim_H'].append(ssim_tensor(x_H, rec_H))
            
            # Visualization (Save first 10 samples)
            if i < 10:
                # Denormalize to [0, 1] numpy
                def to_img(t): return np.clip((t[0,0].cpu().numpy() + 1)/2, 0, 1)
                
                vis_img = np.concatenate([
                    to_img(x_L), to_img(rec_L),  # Low: Input vs Recon
                    to_img(x_H), to_img(rec_H)   # High: Input vs Recon
                ], axis=1) # 拼接到一行
                
                plt.figure(figsize=(10, 3))
                plt.imshow(vis_img, cmap='gray')
                plt.axis('off')
                plt.title(f"Sample {i} | Low(In/Rec) | High(In/Rec)")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"test_sample_{i}.png"))
                plt.close()

    print("\n=== Test Results ===")
    print(f"Low Field  - PSNR: {np.mean(metrics['psnr_L']):.2f} +/- {np.std(metrics['psnr_L']):.2f}")
    print(f"Low Field  - SSIM: {np.mean(metrics['ssim_L']):.4f} +/- {np.std(metrics['ssim_L']):.4f}")
    print(f"High Field - PSNR: {np.mean(metrics['psnr_H']):.2f} +/- {np.std(metrics['psnr_H']):.2f}")
    print(f"High Field - SSIM: {np.mean(metrics['ssim_H']):.4f} +/- {np.std(metrics['ssim_H']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="paired_results")
    parser.add_argument("--sequence_type", type=str, default="all")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    test(args)