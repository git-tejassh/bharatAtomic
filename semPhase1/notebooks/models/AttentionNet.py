import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.rdunet import RDUNet 
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio
from NAFNet_arch import NAFNet, NAFBlock, NAFNetLocal, SimpleGate
from local_arch import AvgPool2d , Local_Base
import arch_util
from pathlib import Path


device = torch.device('mps' if  torch.backends.mps.is_available() else 'cpu')
psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
psnr_metric.to(device)



import os
from PIL import Image
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)   # Before pooling - for skip connection
        down = self.pool(skip) # After pooling - goes deeper
        return skip, down


class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, inter_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.upgrade = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size = 3),
            nn.GroupNorm(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)       # Decoder features
        s1 = self.Ws(s)       # Skip connection features
        out = F.relu(g1 + s1) # Merge signals
        upg = self.upgrade(out)
        upg = F.relu(upg) 
        psi = self.psi(upg)   # Attention map (0 to 1)
        return s * psi        # Filtered skip
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(min(in_channels, skip_channels) // 2, 32)
        self.att = AttentionGate(in_channels, skip_channels, inter_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
  
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        skip = self.att(x, skip)        # Filter skip with attention
        x = torch.cat([x, skip], dim=1) # Merge
        return self.conv(x)
    
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.dec1 = DecoderBlock(512, 256, 256 , inter_channels=256)
        self.dec2 = DecoderBlock(256, 128, 128 , inter_channels=128)
        self.dec3 = DecoderBlock(128, 64, 64 , inter_channels = 64 )

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        b1 = self.bottleneck(p3)
        d1 = self.dec1(b1, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)

        return torch.sigmoid(self.final_conv(d3))