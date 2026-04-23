"""
2D UNet model for attraction field prediction.

Paper 1 uses a 3D VNet-based backbone with two heads (each 6 residual blocks,
kernel 3, padding 1).  Here we adapt to a 2D encoder-decoder UNet which
matches the same two-headed design but uses 2D convolutions.

Outputs
-------
field      : (B, 2, H, W)  raw attraction vectors (dx, dy); no activation
closeness  : (B, 1, H, W)  raw logits (apply sigmoid for probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block with two 3×3 conv layers (as used in VNet heads)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ConvBlock(nn.Module):
    """Standard conv block (no residual) used in encoder/decoder stages."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------
class OutputHead(nn.Module):
    """
    Output head with n_res residual blocks followed by a 1×1 projection.
    Mirrors the VNet head design from Paper 1 (six residual blocks each).
    """

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 6):
        super().__init__()
        layers = [ResBlock(in_ch) for _ in range(n_res)]
        layers.append(nn.Conv2d(in_ch, out_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
class AttractionFieldNet(nn.Module):
    """
    2D UNet backbone with two output heads:
      - field_head    : 2-channel attraction field (dx, dy)
      - closeness_head: 1-channel closeness logit

    Architecture
    ------------
    Encoder: 4 stages of ConvBlock + MaxPool2d(2)
    Bottleneck: ConvBlock
    Decoder: 4 stages of bilinear upsample + cat(skip) + ConvBlock
    Heads: OutputHead (6 ResBlocks + 1×1 conv)

    Default base_ch=32 gives ~7M parameters — lightweight enough for a
    single GPU while matching the two-head design of Paper 1.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32, head_res: int = 6):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        # Encoder
        self.enc1 = ConvBlock(in_ch, ch[0])
        self.enc2 = ConvBlock(ch[0],  ch[1])
        self.enc3 = ConvBlock(ch[1],  ch[2])
        self.enc4 = ConvBlock(ch[2],  ch[3])
        self.bottleneck = ConvBlock(ch[3], ch[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4   = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.dec4  = ConvBlock(ch[3] * 2, ch[3])

        self.up3   = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3  = ConvBlock(ch[2] * 2, ch[2])

        self.up2   = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2  = ConvBlock(ch[1] * 2, ch[1])

        self.up1   = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1  = ConvBlock(ch[0] * 2, ch[0])

        # Two output heads (each with `head_res` residual blocks, mirroring
        # the VNet head design with 6 residual blocks from Paper 1)
        self.field_head    = OutputHead(ch[0], out_ch=2, n_res=head_res)
        self.closeness_head = OutputHead(ch[0], out_ch=1, n_res=head_res)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        field     = self.field_head(d1)      # (B, 2, H, W)
        closeness = self.closeness_head(d1)  # (B, 1, H, W) — raw logits

        return field, closeness
