import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ==========================================
# 1. 路径设置与数据加载器导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
auto_encoder_path = os.path.join(parent_dir, 'auto_encoder')
sys.path.append(auto_encoder_path)

try:
    from data_preprocess import get_ae_dataloader
    print(f"Successfully imported data_preprocess from {auto_encoder_path}")
except ImportError:
    print(f"Error: Could not import data_preprocess from {auto_encoder_path}")
    sys.exit(1)


# ==========================================
# 2. 模型架构定义
# ==========================================

class ContentEncoderVAE(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoderVAE, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

        self.conv_mu = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1, stride=1, bias=True)
        self.conv_logvar = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1, stride=1, bias=True)
        nn.init.constant_(self.conv_logvar.bias, -5.0)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.model(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x): return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return x + self.model(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect': self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate': self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero': self.pad = nn.ZeroPad2d(padding)
        else: assert 0, f"Unsupported padding type: {pad_type}"
        
        if norm == 'bn': self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in': self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'ln': self.norm = LayerNorm(output_dim)
        elif norm == 'adain': self.norm = AdaptiveInstanceNorm2d(output_dim)
        elif norm == 'none': self.norm = None
        else: assert 0, f"Unsupported normalization: {norm}"
        
        if activation == 'relu': self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu': self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh': self.activation = nn.Tanh()
        elif activation == 'none': self.activation = None
        else: assert 0, f"Unsupported activation: {activation}"
        
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=True)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm: x = self.norm(x)
        if self.activation: x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features; self.eps = eps; self.momentum = momentum
        self.weight = None; self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    def forward(self, x):
        assert self.weight is not None and self.bias is not None
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features; self.eps = eps; self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# ==========================================
# 3. Loss 函数
# ==========================================

def kl_pooled_vector(mu, logvar, reduce='mean'):
    mu_vec = mu.mean(dim=[2,3])
    logvar_vec = logvar.mean(dim=[2,3])
    B, C = mu_vec.shape
    kl_per_sample = -0.5 * torch.sum(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1) 
    if reduce == 'mean': return kl_per_sample.sum() / (B * C)
    elif reduce == 'sum': return kl_per_sample.sum()
    else: raise ValueError("reduce must be 'mean' or 'sum'")

def kl_between_gaussians(mu1, logvar1, mu2, logvar2, reduce='mean'):
    if mu1.dim() == 4:
        mu1_v = mu1.mean(dim=[2,3]); logvar1_v = logvar1.mean(dim=[2,3])
        mu2_v = mu2.mean(dim=[2,3]); logvar2_v = logvar2.mean(dim=[2,3])
    else:
        mu1_v, logvar1_v, mu2_v, logvar2_v = mu1, logvar1, mu2, logvar2

    var1 = torch.exp(logvar1_v); var2 = torch.exp(logvar2_v)
    term = logvar2_v - logvar1_v
    term += (var1 + (mu1_v - mu2_v).pow(2)) / (var2 + 1e-8)
    term = 0.5 * (term - 1.0)
    kl_per_sample = term.sum(dim=1)
    
    if reduce == 'mean':
        B, C = mu1_v.shape
        return kl_per_sample.sum() / (B * C)
    elif reduce == 'sum':
        return kl_per_sample.sum()
    else: raise ValueError("reduce must be 'mean' or 'sum'")


# ==========================================
# 4. 训练主逻辑
# ==========================================

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    # 数据加载
    print(f"Loading train/val data from {args.data_root}...")
    train_loader = get_ae_dataloader(
        data_root=args.data_root, sequence_type=args.sequence_type, sequence_types=args.seq_list,
        target_size=(256, 256), split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, preload=args.preload, domain_mode="both"
    )
    val_loader = get_ae_dataloader(
        data_root=args.data_root, sequence_type=args.sequence_type, sequence_types=args.seq_list,
        target_size=(256, 256), split="val", batch_size=args.batch_size,
        num_workers=args.num_workers, preload=args.preload, domain_mode="both", shuffle=False
    )

    # 模型初始化
    enc_params = {'n_downsample': 2, 'n_res': 4, 'input_dim': 1, 'dim': 64, 'norm': 'in', 'activ': 'relu', 'pad_type': 'reflect'}
    dec_params = {'n_upsample': 2, 'n_res': 4, 'dim': 64, 'output_dim': 1, 'res_norm': 'ln', 'activ': 'relu', 'pad_type': 'reflect'}

    E_A = ContentEncoderVAE(**enc_params).to(device)
    G_A = Decoder(**dec_params).to(device)
    E_B = ContentEncoderVAE(**enc_params).to(device)
    G_B = Decoder(**dec_params).to(device)

    all_params = list(E_A.parameters()) + list(G_A.parameters()) + list(E_B.parameters()) + list(G_B.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr)
    recon_loss_fn = nn.L1Loss()

    # 参数
    target_beta = 1e-3
    lambda_align = 1e-2
    warmup_epochs = 5
    anneal_epochs = 20
    best_val_loss = float('inf')

    print(f"Start training for {args.num_epochs} epochs...")

    for epoch in range(1, args.num_epochs + 1):
        E_A.train(); G_A.train(); E_B.train(); G_B.train()
        
        stats = {'rec_L': 0.0, 'rec_H': 0.0, 'kl_prior': 0.0, 'kl_align': 0.0}
        total_samples = 0

        if epoch <= warmup_epochs: beta = 0.0
        elif epoch <= warmup_epochs + anneal_epochs: beta = target_beta * ((epoch - warmup_epochs) / anneal_epochs)
        else: beta = target_beta

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.num_epochs} [Train] beta={beta:.5f}")
        
        for batch in pbar:
            images = batch['images']
            B = images.size(0)
            x_L = images[:, 0].to(device) * 2.0 - 1.0
            x_H = images[:, 1].to(device) * 2.0 - 1.0

            optimizer.zero_grad()

            z_L, mu_L, logvar_L = E_A(x_L); rec_L = G_A(z_L)
            loss_rec_L = recon_loss_fn(rec_L, x_L)

            z_H, mu_H, logvar_H = E_B(x_H); rec_H = G_B(z_H)
            loss_rec_H = recon_loss_fn(rec_H, x_H)

            loss_kl_prior = kl_pooled_vector(mu_L, logvar_L) + kl_pooled_vector(mu_H, logvar_H)
            loss_kl_align = kl_between_gaussians(mu_L, logvar_L, mu_H, logvar_H)

            loss = (loss_rec_L + loss_rec_H) + beta * loss_kl_prior + lambda_align * loss_kl_align

            loss.backward()
            optimizer.step()

            stats['rec_L'] += loss_rec_L.item() * B
            stats['rec_H'] += loss_rec_H.item() * B
            stats['kl_prior'] += loss_kl_prior.item() * B
            stats['kl_align'] += loss_kl_align.item() * B
            total_samples += B
            
            # === 修改处：进度条显示所有指标 ===
            pbar.set_postfix({
                'rL': f"{loss_rec_L.item():.4f}",
                'rH': f"{loss_rec_H.item():.4f}",
                'kl': f"{loss_kl_prior.item():.4f}",
                'ali': f"{loss_kl_align.item():.4f}",
                'tot': f"{loss.item():.4f}"
            })

        for k in stats: stats[k] /= total_samples
        writer.add_scalar('Train/Rec_Low', stats['rec_L'], epoch)
        writer.add_scalar('Train/Rec_High', stats['rec_H'], epoch)
        writer.add_scalar('Train/KL_Align', stats['kl_align'], epoch)
        writer.add_scalar('Train/KL_Prior', stats['kl_prior'], epoch)
        writer.add_scalar('Train/Beta', beta, epoch)

        # Val
        E_A.eval(); G_A.eval(); E_B.eval(); G_B.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch}/{args.num_epochs} [Val]"):
                images = batch['images']
                B = images.size(0)
                x_L = images[:, 0].to(device) * 2.0 - 1.0
                x_H = images[:, 1].to(device) * 2.0 - 1.0
                
                z_L, _, _ = E_A(x_L); rec_L = G_A(z_L)
                z_H, _, _ = E_B(x_H); rec_H = G_B(z_H)
                
                loss_val_batch = recon_loss_fn(rec_L, x_L) + recon_loss_fn(rec_H, x_H)
                val_loss += loss_val_batch.item() * B
                val_samples += B
        
        avg_val_loss = val_loss / val_samples
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        
        # === 修改处：每轮打印所有指标 ===
        print(f"Epoch {epoch}: RecL={stats['rec_L']:.4f} RecH={stats['rec_H']:.4f} KL={stats['kl_prior']:.4f} Align={stats['kl_align']:.4f} | ValLoss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = os.path.join(ckpt_dir, "best_paired_model.pth")
            torch.save({
                'epoch': epoch, 'E_A': E_A.state_dict(), 'G_A': G_A.state_dict(),
                'E_B': E_B.state_dict(), 'G_B': G_B.state_dict(),
                'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss
            }, path)
            print(f"  --> Saved Best Model to {path}")
            
        if epoch % 20 == 0:
            path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'E_A': E_A.state_dict(), 'G_A': G_A.state_dict()}, path)

    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/data54/wanghaobo/data/Third_full/")
    parser.add_argument("--out_dir", type=str, default="paired_results")
    parser.add_argument("--sequence_type", type=str, default="all")
    parser.add_argument("--seq_list", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--preload", action='store_true')
    
    args = parser.parse_args()
    train(args)