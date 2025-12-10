import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)

class AnatomyLoss(nn.Module):
    def __init__(self, mode='l1'):
        super().__init__()
        if mode == 'l1':
            self.criterion = nn.L1Loss()
        elif mode == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, img_gen, img_source):
        return self.criterion(img_gen, img_source)

class StyleLoss(nn.Module):
    """
    Loss to encourage style difference (e.g. contrast, SNR).
    User requirement: "使信噪比，对比度等发生变化，这个差异要尽可能大"
    We can maximize the difference in mean and std deviation.
    """
    def __init__(self, maximize=True):
        super().__init__()
        self.maximize = maximize

    def compute_stats(self, x):
        # x: (B, C, H, W)
        # Compute mean and std per instance
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return mean, std

    def forward(self, img_gen, img_source):
        gen_mean, gen_std = self.compute_stats(img_gen)
        src_mean, src_std = self.compute_stats(img_source)

        # Simple L2 distance between statistics
        loss_mean = F.mse_loss(gen_mean, src_mean)
        loss_std = F.mse_loss(gen_std, src_std)
        
        total_diff = loss_mean + loss_std

        if self.maximize:
            # We want to maximize difference, so we minimize -difference
            # Add a small epsilon to avoid numerical issues if needed, or just return negative
            return -total_diff
        else:
            return total_diff

class ContrastLoss(nn.Module):
    """
    Specific loss for contrast difference.
    Contrast can be approximated by standard deviation.
    """
    def __init__(self, maximize=True):
        super().__init__()
        self.maximize = maximize

    def forward(self, img_gen, img_source):
        gen_std = img_gen.std(dim=[1, 2, 3])
        src_std = img_source.std(dim=[1, 2, 3])
        
        diff = torch.abs(gen_std - src_std).mean()
        
        if self.maximize:
            return -diff
        else:
            return diff
