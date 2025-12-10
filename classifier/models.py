import torch
import torch.nn as nn

class LatentClassifier(nn.Module):
    """
    Classifier that takes the latent code z from AE and predicts the domain (Low/High).
    Input z shape: (B, C, H, W). For AE_Vanilla default: (B, 128, 16, 16).
    """
    def __init__(self, in_channels=128, num_classes=2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, z):
        return self.net(z)
