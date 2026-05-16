import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ==================================================
# Conv Block (GroupNorm)
# ==================================================

def conv_block_gn(in_c, out_c, dropout=False):

    layers = [

        nn.Conv2d(
            in_c,
            out_c,
            3,
            padding=1,
            bias=False,
        ),

        nn.GroupNorm(8, out_c),

        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_c,
            out_c,
            3,
            padding=1,
            bias=False,
        ),

        nn.GroupNorm(8, out_c),

        nn.ReLU(inplace=True),
    ]

    if dropout:
        layers.append(nn.Dropout2d(0.1))

    return nn.Sequential(*layers)


# ==================================================
# Learnable DownSample
# ==================================================

class Down(nn.Module):

    def __init__(self, in_c, out_c, dropout=False):

        super().__init__()

        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.block = conv_block_gn(
            out_c,
            out_c,
            dropout,
        )

    def forward(self, x):

        x = self.conv(x)

        return self.block(x)


# ==================================================
# Anti-checkerboard UpSample
# ==================================================

class Up(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.up = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),

            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(8, out_c),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.up(x)


# ==================================================
# Smooth UNet for ILT
# ==================================================

@register_model("unet_smooth")
class UNetSmooth(nn.Module):

    def __init__(self, base_c=32):

        super().__init__()

        # ==========================================
        # Encoder
        # ==========================================

        self.enc1 = conv_block_gn(
            1,
            base_c,
        )

        self.down1 = Down(
            base_c,
            base_c * 2,
        )

        self.enc2 = conv_block_gn(
            base_c * 2,
            base_c * 2,
        )

        self.down2 = Down(
            base_c * 2,
            base_c * 4,
        )

        self.enc3 = conv_block_gn(
            base_c * 4,
            base_c * 4,
            dropout=True,
        )

        self.down3 = Down(
            base_c * 4,
            base_c * 8,
            dropout=True,
        )

        # ==========================================
        # Bottleneck
        # ==========================================

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                base_c * 8,
                base_c * 8,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),

            nn.GroupNorm(8, base_c * 8),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_c * 8,
                base_c * 8,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),

            nn.GroupNorm(8, base_c * 8),

            nn.ReLU(inplace=True),
        )

        # ==========================================
        # Decoder
        # ==========================================

        self.up3 = Up(
            base_c * 8,
            base_c * 4,
        )

        self.dec3 = conv_block_gn(
            base_c * 8,
            base_c * 4,
        )

        self.up2 = Up(
            base_c * 4,
            base_c * 2,
        )

        self.dec2 = conv_block_gn(
            base_c * 4,
            base_c * 2,
        )

        self.up1 = Up(
            base_c * 2,
            base_c,
        )

        self.dec1 = conv_block_gn(
            base_c * 2,
            base_c,
        )

        # ==========================================
        # Output
        # ==========================================

        self.out = nn.Conv2d(
            base_c,
            1,
            kernel_size=1,
        )
        
        # ==========================================
        # Zero Initialization
        # Initial delta ≈ 0
        # ==========================================

        nn.init.zeros_(self.out.weight)

        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):

        # ==========================================
        # Encoder
        # ==========================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.down1(e1)
        )

        e3 = self.enc3(
            self.down2(e2)
        )

        # ==========================================
        # Bottleneck
        # ==========================================

        b = self.bottleneck(
            self.down3(e3)
        )

        # ==========================================
        # Decoder
        # ==========================================

        d3 = self.up3(b)

        d3 = self.dec3(
            torch.cat([d3, e3], dim=1)
        )

        d2 = self.up2(d3)

        d2 = self.dec2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = self.up1(d2)

        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )

        # ==========================================
        # Output
        # ==========================================

        return self.out(d1)