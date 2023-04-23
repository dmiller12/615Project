import torch

from escnn import gspaces
from escnn import nn


class SO2SteerableCNN(torch.nn.Module):

    def __init__(self, n_classes=10, n_channels=1):

        super(SO2SteerableCNN, self).__init__()

        self.r2_act = gspaces.rot2dOnR2(N=-1)

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*n_channels)

        self.input_type = in_type

        self.mask = nn.MaskModule(in_type, 29, margin=1)

        activation1 = nn.FourierELU(self.r2_act, 24, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        in_type = self.block1.out_type
        activation2 = nn.FourierELU(self.r2_act, 48, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation2
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block2.out_type
        activation3 = nn.FourierELU(self.r2_act, 48, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation3
        )

        in_type = self.block3.out_type
        activation4 = nn.FourierELU(self.r2_act, 96, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation4
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block4.out_type
        activation5 = nn.FourierELU(self.r2_act, 96, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation5
        )

        in_type = self.block5.out_type
        activation6 = nn.FourierELU(self.r2_act, 64, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        c = 64

        output_invariant_type = nn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor, rot=None):
        x = self.input_type(input)

        x = self.mask(x)

        first = self.block1(x)
        x = self.block2(first)
        x = self.pool1(x)

        x = self.block3(x)
        mid_feats = self.block4(x)
        x = self.pool2(mid_feats)

        x = self.block5(x)
        x = self.block6(x)

        x = self.pool3(x)

        x = self.invariant_map(x)

        x = x.tensor

        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x