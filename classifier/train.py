import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add auto-encoder to path to import AE_Vanilla and AEImageDataset
ae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../auto_encoder'))
sys.path.append(ae_path)
print(f"Added to sys.path: {ae_path}")
print(f"Contents: {os.listdir(ae_path)}")

try:
    from train_encoder import AE_Vanilla
    from data_preprocess import AEImageDataset
except ImportError as e:
    print(f"Error importing from auto_encoder: {e}")
    import traceback
    traceback.print_exc()
    print("Please ensure 'train_encoder.py' and 'data_preprocess.py' are in '../auto_encoder'")
    sys.exit(1)

from models import LatentClassifier
from losses import ClassificationLoss

def train_classifier(
    data_root,
    ae_checkpoint,
    out_dir,
    batch_size=32,
    num_epochs=50,
    lr=1e-4,
    gpu_id=0,
    hidden_dims=[16, 32, 64, 128],
    target_size=(256, 256)
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'logs'))

    # 1. Load Data
    # We need both domains to train the classifier
    train_dataset = AEImageDataset(
        data_root=data_root,
        split='train',
        domain_mode='both',
        target_size=target_size
    )
    val_dataset = AEImageDataset(
        data_root=data_root,
        split='val',
        domain_mode='both',
        target_size=target_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    # 2. Load Pre-trained AE
    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        checkpoint = torch.load(ae_checkpoint, map_location=device)
        
        if 'hidden_dims' in checkpoint:
            hidden_dims = checkpoint['hidden_dims']
            print(f"Overwriting hidden_dims from checkpoint: {hidden_dims}")
            
        ae = AE_Vanilla(in_channels=1, hidden_dims=hidden_dims).to(device)

        # Handle if checkpoint has 'model_state_dict' or 'model_state' or is just state_dict
        if 'model_state_dict' in checkpoint:
            ae.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            ae.load_state_dict(checkpoint['model_state'])
        else:
            ae.load_state_dict(checkpoint)
    else:
        print(f"Warning: AE checkpoint not found at {ae_checkpoint}. Using random init (NOT RECOMMENDED).")
        ae = AE_Vanilla(in_channels=1, hidden_dims=hidden_dims).to(device)
    
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    # 3. Initialize Classifier
    # Latent shape: (B, 128, 16, 16) for default hidden_dims
    classifier = LatentClassifier(in_channels=hidden_dims[-1], num_classes=2).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = ClassificationLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # images: (B, K, 1, H, W), domains: (B, K)
            images = batch['images']
            domains = batch['domains']
            
            # Flatten B and K
            B, K, C, H, W = images.shape
            images = images.view(B*K, C, H, W)
            domains = domains.view(B*K)

            images = images.to(device)
            # Map domains to long for CrossEntropy: 0 or 1
            labels = domains.long().to(device)

            # Encode
            with torch.no_grad():
                # AE expects [-1, 1]
                images_norm = images * 2 - 1
                z = ae.encoder(images_norm)

            # Predict
            outputs = classifier(z)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

        avg_train_loss = train_loss / total
        train_acc = correct / total
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Validation
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = batch['images']
                domains = batch['domains']
                
                B, K, C, H, W = images.shape
                images = images.view(B*K, C, H, W)
                domains = domains.view(B*K)

                images = images.to(device)
                labels = domains.long().to(device)

                images_norm = images * 2 - 1
                z = ae.encoder(images_norm)
                outputs = classifier(z)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / total
        val_acc = correct / total
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")

        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(out_dir, 'best_classifier.pth')
            torch.save(classifier.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, os.path.join(out_dir, 'checkpoint.pth'))

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to pretrained AE checkpoint")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    
    train_classifier(
        data_root=args.data_root,
        ae_checkpoint=args.ae_checkpoint,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gpu_id=args.gpu_id
    )
