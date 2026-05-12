import torch
import torch.nn as nn
from .registry import register_model


# =========================
# 简洁 Conv Block（便于单图过拟合）
# =========================
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# =========================
# 下采样
# =========================
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = conv_block(in_c, out_c)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)


# =========================
# 上采样
# =========================
class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.block = conv_block(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# =========================
# 主模型
# =========================
@register_model("unet_test")
class UNetTest(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()

        # encoder
        self.enc1 = conv_block(1, base_c)
        self.enc2 = Down(base_c, base_c * 2)
        self.enc3 = Down(base_c * 2, base_c * 4)

        # bottleneck
        self.bottleneck = Down(base_c * 4, base_c * 8)

        # decoder
        self.up3 = Up(base_c * 8, base_c * 4, base_c * 4)
        self.up2 = Up(base_c * 4, base_c * 2, base_c * 2)
        self.up1 = Up(base_c * 2, base_c, base_c)

        # output logits
        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out(d1)