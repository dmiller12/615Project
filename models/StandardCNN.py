import torch

from torch import nn
import torch.nn.functional as F

class StandardCNN(torch.nn.Module):

    def __init__(self, n_classes=10, n_channels=1) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, 24, kernel_size=7, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.block5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=5, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)

        self.fully_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_classes),
        )


    def forward(self, x, rot=None):

        first = self.block1(x)
        x = self.block2(first)
        x = self.pool1(x)

        x = self.block3(x)
        mid_feats = self.block4(x)
        x = self.pool2(mid_feats)

        x = self.block5(x)
        x = self.block6(x)

        x = self.pool3(x)

        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x
