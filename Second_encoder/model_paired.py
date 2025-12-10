import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
#  工具函数：KL 散度计算
# ==========================================

def kl_pooled_vector(mu, logvar, reduce='mean'):
    """
    计算单个分布与标准正态分布 N(0, I) 的 KL 散度。
    先在空间维度 (H, W) 做平均池化，得到 (B, C) 向量，再计算 KL。
    """
    # mu, logvar: (B, C, H, W)
    mu_vec = mu.mean(dim=[2, 3])          # (B, C)
    logvar_vec = logvar.mean(dim=[2, 3])  # (B, C)
    B, C = mu_vec.shape
    
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_sample = -0.5 * torch.sum(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1) # (B,)
    
    if reduce == 'mean':
        return kl_per_sample.sum() / (B * C) # 平均到每个维度
    elif reduce == 'sum':
        return kl_per_sample.sum()
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")

def kl_between_gaussians(mu1, logvar1, mu2, logvar2, reduce='mean'):
    """
    计算两个高斯分布之间的 KL 散度：KL( N(mu1, var1) || N(mu2, var2) )
    用于强制两个编码器的潜在空间对齐。
    """
    # 先做空间池化
    if mu1.dim() == 4:
        mu1_v = mu1.mean(dim=[2, 3])
        logvar1_v = logvar1.mean(dim=[2, 3])
        mu2_v = mu2.mean(dim=[2, 3])
        logvar2_v = logvar2.mean(dim=[2, 3])
    else:
        mu1_v, logvar1_v, mu2_v, logvar2_v = mu1, logvar1, mu2, logvar2

    var1 = torch.exp(logvar1_v)
    var2 = torch.exp(logvar2_v)

    # 公式: log(std2/std1) + (var1 + (mu1-mu2)^2) / (2*var2) - 0.5
    # log(std2/std1) = 0.5 * (logvar2 - logvar1)
    term = logvar2_v - logvar1_v
    term += (var1 + (mu1_v - mu2_v).pow(2)) / (var2 + 1e-8)
    term = 0.5 * (term - 1.0) 
    
    kl_per_sample = term.sum(dim=1) # (B,)
    
    if reduce == 'mean':
        B, C = mu1_v.shape
        return kl_per_sample.sum() / (B * C)
    elif reduce == 'sum':
        return kl_per_sample.sum()
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")

# ==========================================
#  网络架构 (复现)
# ==========================================

class ContentEncoderVAE(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoderVAE, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # 下采样
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # 残差块
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

        # 均值和方差预测层
        self.conv_mu = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1, stride=1, bias=True)
        self.conv_logvar = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1, stride=1, bias=True)

        # 初始化 logvar 为较小值
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
        # 残差块
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # 上采样
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # 输出层 (Tanh -> [-1, 1])
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

# --- 基础模块 (Block, ResBlock 等) ---
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
        # Padding
        if pad_type == 'reflect': self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate': self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero': self.pad = nn.ZeroPad2d(padding)
        else: assert 0, f"Unsupported padding type: {pad_type}"
        # Norm
        if norm == 'bn': self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in': self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'ln': self.norm = LayerNorm(output_dim)
        elif norm == 'adain': self.norm = AdaptiveInstanceNorm2d(output_dim)
        elif norm == 'none': self.norm = None
        else: assert 0, f"Unsupported normalization: {norm}"
        # Activation
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
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
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
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
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