import argparse
import os
import sys
import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

# Add auto-encoder to path
ae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../auto_encoder'))
sys.path.append(ae_path)
print(f"Added to sys.path: {ae_path}")
print(f"Contents of {ae_path}: {os.listdir(ae_path)}")

try:
    from train_encoder import AE_Vanilla
    from data_preprocess import AEImageDataset
except ImportError as e:
    print(f"Error importing from auto_encoder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from models import LatentClassifier
from losses import ClassificationLoss, AnatomyLoss, StyleLoss

def infer_counterfactual(
    data_root,
    ae_checkpoint,
    classifier_checkpoint,
    out_dir,
    target_domain='high', # 'low' or 'high'
    num_steps=100,
    lr=1e-2,
    lambda_cls=1.0,
    lambda_anat=1.0,
    lambda_style=1.0,
    gpu_id=0,
    hidden_dims=[16, 32, 64, 128],
    num_samples=5
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Models
    print(f"Loading AE checkpoint from {ae_checkpoint}")
    checkpoint = torch.load(ae_checkpoint, map_location=device)
    
    if 'hidden_dims' in checkpoint:
        hidden_dims = checkpoint['hidden_dims']
        print(f"Overwriting hidden_dims from checkpoint: {hidden_dims}")

    ae = AE_Vanilla(in_channels=1, hidden_dims=hidden_dims).to(device)
    
    if 'model_state_dict' in checkpoint:
        ae.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        ae.load_state_dict(checkpoint['model_state'])
    else:
        ae.load_state_dict(checkpoint)
    
    ae.eval()
    for p in ae.parameters(): p.requires_grad = False

    # Try to load classifier with correct dims
    try:
        classifier = LatentClassifier(in_channels=hidden_dims[-1], num_classes=2).to(device)
        classifier.load_state_dict(torch.load(classifier_checkpoint, map_location=device))
        print("Successfully loaded classifier with matching dimensions.")
    except RuntimeError as e:
        print(f"Warning: Failed to load classifier with hidden_dims={hidden_dims[-1]}. Error: {e}")
        print("Attempting to fallback to default hidden_dims=[..., 128] for classifier and AE.")
        
        # Fallback: assume classifier was trained with default dims (128)
        # This means we must use an AE with 128 dims too.
        # But the AE checkpoint is 16 dims. So we can't load the AE weights correctly.
        # We will load AE with 128 dims and try to load weights (strict=False) or just random init.
        
        fallback_hidden_dims = [16, 32, 64, 128]
        ae = AE_Vanilla(in_channels=1, hidden_dims=fallback_hidden_dims).to(device)
        
        # Try to load AE weights partially
        try:
            if 'model_state_dict' in checkpoint:
                ae.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'model_state' in checkpoint:
                ae.load_state_dict(checkpoint['model_state'], strict=False)
            else:
                ae.load_state_dict(checkpoint, strict=False)
            print("Loaded AE weights with strict=False (expect size mismatches).")
        except Exception as ae_e:
            print(f"Could not load AE weights: {ae_e}. Using random initialization.")
            
        ae.eval()
        for p in ae.parameters(): p.requires_grad = False
        
        # Now load classifier with 128 dims
        classifier = LatentClassifier(in_channels=128, num_classes=2).to(device)
        classifier.load_state_dict(torch.load(classifier_checkpoint, map_location=device))
        print("Successfully loaded classifier with fallback dimensions (128).")
        print("WARNING: Results may be invalid due to AE/Classifier dimension mismatch during training.")

    classifier.eval()
    for p in classifier.parameters(): p.requires_grad = False

    # 2. Load Data
    # If target is high, we want input from low
    source_domain = 'low' if target_domain == 'high' else 'high'
    dataset = AEImageDataset(
        data_root=data_root,
        split='val', # Use validation set for inference
        domain_mode=source_domain,
        target_size=(256, 256)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # 3. Losses
    criterion_cls = ClassificationLoss()
    criterion_anat = AnatomyLoss(mode='l1')
    criterion_style = StyleLoss(maximize=True)

    target_label_idx = 1 if target_domain == 'high' else 0
    
    print(f"Starting inference: Source={source_domain}, Target={target_domain} (Label {target_label_idx})")

    count = 0
    for i, batch in enumerate(loader):
        if count >= num_samples:
            break
        
        img_source = batch['images'] # (B, K, 1, H, W)
        # Since domain_mode is 'low' or 'high', K=1
        B, K, C, H, W = img_source.shape
        img_source = img_source.view(B*K, C, H, W)
        
        img_source = img_source.to(device)
        
        # Normalize to [-1, 1] for AE
        img_source_norm = img_source * 2 - 1
        
        # Encode
        with torch.no_grad():
            z_init = ae.encoder(img_source_norm)
        
        # Optimize z
        z_opt = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_opt], lr=lr)
        
        target_tensor = torch.tensor([target_label_idx], device=device).long()

        pbar = tqdm(range(num_steps), desc=f"Sample {count+1}")
        for step in pbar:
            optimizer.zero_grad()
            
            # Decode
            img_gen_norm = ae.decode(z_opt)
            # img_gen is in [-1, 1]
            
            # Predict
            pred = classifier(z_opt)
            
            # Losses
            # 1. Classification: make it look like target domain
            loss_cls = criterion_cls(pred, target_tensor)
            
            # 2. Anatomy: keep structure similar to source
            # We compare in normalized space [-1, 1] or [0, 1]? 
            # img_source_norm is [-1, 1].
            loss_anat = criterion_anat(img_gen_norm, img_source_norm)
            
            # 3. Style: maximize difference from source style
            loss_style = criterion_style(img_gen_norm, img_source_norm)
            
            total_loss = lambda_cls * loss_cls + lambda_anat * loss_anat + lambda_style * loss_style
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'L_cls': loss_cls.item(),
                'L_anat': loss_anat.item(),
                'L_style': loss_style.item()
            })

        # Final generation
        with torch.no_grad():
            img_cf_norm = ae.decode(z_opt)
            
            # Denormalize to [0, 1] for visualization
            img_source_vis = (img_source_norm + 1) / 2
            img_cf_vis = (img_cf_norm + 1) / 2
            diff = torch.abs(img_source_vis - img_cf_vis)

            # Concatenate: Source | Counterfactual | Difference
            res = torch.cat([img_source_vis, img_cf_vis, diff], dim=3)
            save_path = os.path.join(out_dir, f"sample_{count}_{source_domain}2{target_domain}.png")
            vutils.save_image(res, save_path)
            print(f"Saved result to {save_path}")
        
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--classifier_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/inference")
    parser.add_argument("--target_domain", type=str, default="high", choices=["low", "high"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    
    args = parser.parse_args()
    
    infer_counterfactual(
        data_root=args.data_root,
        ae_checkpoint=args.ae_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        out_dir=args.out_dir,
        target_domain=args.target_domain,
        gpu_id=args.gpu_id,
        num_samples=args.num_samples
    )
