import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


# =========================
# 更稳定的Conv Block（GN替代BN）
# =========================
def conv_block_gn(in_c, out_c, dropout=False):
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_c),   # ✅ 替换BN
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.1))
    return nn.Sequential(*layers)


# =========================
# 可学习下采样（替代MaxPool）
# =========================
class Down(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)  # ✅ stride=2
        self.block = conv_block_gn(out_c, out_c, dropout)

    def forward(self, x):
        x = self.conv(x)
        return self.block(x)


# =========================
# 主模型
# =========================
@register_model("unet_edge_dilated_64")
class UNetEdge_64_v2(nn.Module):
    def __init__(self, base_c=64):
        super().__init__()

        # Encoder
        self.enc1 = conv_block_gn(1, base_c)
        self.down1 = Down(base_c, base_c * 2)

        self.enc2 = conv_block_gn(base_c * 2, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)

        self.enc3 = conv_block_gn(base_c * 4, base_c * 4, dropout=True)
        self.down3 = Down(base_c * 4, base_c * 8, dropout=True)

        # Bottleneck（加入 dilation 提升感受野）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_c * 8, base_c * 8, 3, padding=2, dilation=2),
            nn.GroupNorm(8, base_c * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_c * 8, base_c * 8, 3, padding=2, dilation=2),
            nn.GroupNorm(8, base_c * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder（基本不动，保持稳定性）
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = conv_block_gn(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = conv_block_gn(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = conv_block_gn(base_c * 2, base_c)

        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)