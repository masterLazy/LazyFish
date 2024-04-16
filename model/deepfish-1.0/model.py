# U-Net Model

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, width):
        super(Model, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((8,13,13)),
            
            nn.Conv2d(8, 16, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((16,11,11)),
            
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((32,9,9)),
            
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((64,7,7)),
            
            nn.ConvTranspose2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((32,9,9)),
            
            nn.ConvTranspose2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((16,11,11)),
            
            nn.ConvTranspose2d(16, 8, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((8,13,13)),
            
            nn.ConvTranspose2d(8, 1, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((1,15,15))
        )

    def forward(self, x):
        return self.seq(x)
