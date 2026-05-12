import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

def conv_block(in_c, out_c, dropout=False):
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True)
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.2))
    return nn.Sequential(*layers)



@register_model("unet_full")
class UNetFull(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre = nn.Conv2d(1, 32, kernel_size=15, padding=7)

        self.enc1 = conv_block(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256, dropout=True)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512, dropout=True)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)
        
    
    def forward(self, x):
        x = self.pre(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)