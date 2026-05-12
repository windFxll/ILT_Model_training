import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

def conv_block_bn(in_c, out_c, dropout=False):
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.1))  # 降低一点
    return nn.Sequential(*layers)

@register_model("unet_edge_64")
class UNetEdge_64(nn.Module):
    def __init__(self, base_c=64):
        super().__init__()

        self.enc1 = conv_block_bn(1, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block_bn(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block_bn(base_c * 2, base_c * 4, dropout=True)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block_bn(base_c * 4, base_c * 8, dropout=True)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = conv_block_bn(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = conv_block_bn(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = conv_block_bn(base_c * 2, base_c)
        
        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
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

        return self.out(d1)  # logits