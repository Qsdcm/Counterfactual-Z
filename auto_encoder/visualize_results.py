import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from train_encoder import AE_Vanilla
from data_preprocess import get_ae_dataloader

def visualize(args):
    # Device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Checkpoint
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        return

    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # Get config from checkpoint if available, otherwise use args or defaults
    hidden_dims = checkpoint.get("hidden_dims", [16, 32, 64, 128])
    print(f"Model hidden_dims: {hidden_dims}")

    # Initialize Model
    model = AE_Vanilla(in_channels=1, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Output directories
    recon_dir = os.path.join("ae_results", "recon")
    low_dir = os.path.join(recon_dir, "low")
    high_dir = os.path.join(recon_dir, "high")
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)
    print(f"Saving results to {recon_dir}")

    # Data Loader
    # We use 'test' split for visualization
    test_loader = get_ae_dataloader(
        data_root=args.data_root,
        sequence_type=args.sequence_type,
        target_size=(args.target_h, args.target_w),
        split="test",
        batch_size=args.num_samples,
        num_workers=4,
        preload=False, # No need to preload for just a few samples
        domain_mode=args.domain_mode,
        shuffle=True # Shuffle to get random samples
    )

    # Get a batch
    try:
        batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Test dataset is empty.")
        return

    images = batch["images"]  # (B, K, 1, H, W)
    domains = batch["domains"] # (B, K)
    B, K, C, H, W = images.shape
    
    # Flatten to (B*K, 1, H, W)
    images_flat = images.view(B * K, C, H, W).to(device)
    domains_flat = domains.view(B * K)
    
    # Normalize to [-1, 1] for model input
    imgs_in = images_flat * 2.0 - 1.0

    # Inference
    with torch.no_grad():
        recon = model(imgs_in)

    # Denormalize to [0, 1] for display
    imgs_in_disp = images_flat.cpu().numpy()
    imgs_out_disp = (recon.cpu().numpy() + 1.0) / 2.0
    imgs_out_disp = np.clip(imgs_out_disp, 0, 1)

    # Plotting
    # We will plot num_samples pairs
    n = B * K
    
    for i in range(n):
        img_in = imgs_in_disp[i, 0]
        img_out = imgs_out_disp[i, 0]
        domain_val = domains_flat[i].item()

        # Determine save path based on domain
        if domain_val < 0.5: # Low field (0.0)
            save_dir = low_dir
            prefix = "low"
        else: # High field (1.0)
            save_dir = high_dir
            prefix = "high"

        plt.figure(figsize=(6, 3))
        
        # Input
        plt.subplot(1, 2, 1)
        plt.imshow(img_in, cmap='gray', vmin=0, vmax=1)
        plt.title("Input")
        plt.axis('off')

        # Reconstruction
        plt.subplot(1, 2, 2)
        plt.imshow(img_out, cmap='gray', vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.axis('off')

        save_path = os.path.join(save_dir, f"{prefix}_sample_{i}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"Visualization saved to {recon_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best_ae.pth")
    parser.add_argument("--data_root", type=str, default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--sequence_type", type=str, default="all")
    parser.add_argument("--domain_mode", type=str, default="both")
    parser.add_argument("--target_h", type=int, default=256)
    parser.add_argument("--target_w", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=5, help="Number of image pairs to show")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    visualize(args)