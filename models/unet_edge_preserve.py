import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)
        self.block = ResBlock(out_c, out_c)

    def forward(self, x):
        x = self.conv(x)
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.block = ResBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        # 对齐尺寸（防止奇数尺寸问题）
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.block(x)

@register_model("unet_edge_preserve")
class UNetEdgePreserve(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()

        # ---- Encoder（减少压缩）----
        self.enc1 = ResBlock(1, base_c)          # H
        self.down1 = Down(base_c, base_c * 2)    # H/2
        self.down2 = Down(base_c * 2, base_c * 4)  # H/4

        # ❗不再继续下采样（避免 H/8）

        # ---- Bottleneck ----
        self.bottleneck = ResBlock(base_c * 4, base_c * 4)

        # ---- Decoder ----
        self.up2 = Up(base_c * 4, base_c * 2)
        self.up1 = Up(base_c * 2, base_c)

        # ---- 输出层（强化浅层信息）----
        self.out = nn.Sequential(
            nn.Conv2d(base_c, base_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, 1, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # H
        e2 = self.down1(e1)      # H/2
        e3 = self.down2(e2)      # H/4

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d2 = self.up2(b, e2)
        d1 = self.up1(d2, e1)

        return self.out(d1)