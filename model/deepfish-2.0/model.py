# DeepFish Model

import torch
from torch import nn


# (conv3x3 + ReLU) x2
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)


# Downsample
# pool2x2 + ConvBlock
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.seq(x)


# Upsample
# up-conv2x2 + cat + ConvBlock
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, output_padding=1
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, cat):
        # up-conv2x2
        x = self.up_conv(x)
        # cat
        x = torch.cat((x, cat), dim=1)
        # conv
        x = self.conv(x)
        return x


# U-Net
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # First
        self.fr = ConvBlock(1, 8)

        # DownSample
        self.down = nn.ModuleList()
        self.down.append(DownSample(8, 16))
        self.down.append(DownSample(16, 32))
        # Upsample
        self.up = nn.ModuleList()
        self.up.append(UpSample(32, 16))
        self.up.append(UpSample(16, 8))
        # Final
        self.fl = nn.Conv2d(8, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = []
        x1 = self.fr(x)
        x2 = self.down[0](x1)
        x = self.down[1](x2)
        x = self.up[0](x, x2)
        x = self.up[1](x, x1)
        x = self.fl(x)
        b, c, h, w = x.size()
        x = self.softmax(x.view(b, -1)).view(b, c, h, w)
        return x
