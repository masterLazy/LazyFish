# DeepFish Model

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, width):
        super(Model, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.LeakyReLU(),
            nn.LayerNorm((4, 11, 11)),
            #
            nn.Conv2d(4, 8, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((8, 9, 9)),
            #
            nn.Conv2d(8, 16, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((16, 7, 7)),
        )
        self.fnn = nn.Sequential(
            nn.Linear(16 * 7 * 7, 16 * 7 * 7 * 4),
            nn.LeakyReLU(),
            nn.LayerNorm((16 * 7 * 7 * 4)),
            #
            nn.Linear(16 * 7 * 7 * 4, 16 * 7 * 7),
            nn.LeakyReLU(),
            nn.LayerNorm((16 * 7 * 7)),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((8, 9, 9)),
            #
            nn.ConvTranspose2d(8, 4, 3),
            nn.LeakyReLU(),
            nn.LayerNorm((4, 11, 11)),
            #
            nn.ConvTranspose2d(4, 1, 5),
            nn.LeakyReLU(),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down(x)
        b, c, h, w = x.size()
        x = x.view(b, -1)
        x = x + self.fnn(x)
        x = self.up(x.view(b, c, h, w))
        b, c, h, w = x.size()
        x = self.softmax(x.view(b, -1)).view(b, c, h, w)
        return x
