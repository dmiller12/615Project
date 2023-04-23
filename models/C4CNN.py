import numpy as np
import torch
from torch import nn


def rotate_p4(y: torch.Tensor, r: int) -> torch.Tensor:
    assert len(y.shape) >= 3
    assert y.shape[-3] == 4


    r_inv = (4 - r) % 4
    permute = torch.arange(4)
    permute = (permute + r_inv) % 4

    y = y.rot90(r, dims=(-2, -1))

    return y[:, :, permute, :, :]



class LiftingConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):

        super(LiftingConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.weight = None

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        self.weight = torch.nn.Parameter(w, requires_grad=True)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)



    def build_filter(self) -> torch.Tensor:
        n_out, n_in, n_k, n_k = self.weight.shape
        self.register_buffer("_filter", torch.empty((n_out, 4, n_in, n_k, n_k)), persistent=False)

        for i in range(4):
            self._filter[:, i, :, :, :] = self.weight.rot90(i, dims=(-2, -1))

        self.register_buffer("_bias", torch.empty((self.bias.shape[0], 4)), persistent=False)
        if self.bias is not None:
            for i in range(4):
                self._bias[:, i] = self.bias

        else:
            _bias = None

        device = self.weight.device
        return self._filter.to(device), self._bias.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * 4)

        out = torch.conv2d(x, _filter, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=_bias)

        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])


class GroupConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):

        super(GroupConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels


        w = torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        self.weight = torch.nn.Parameter(w, requires_grad=True)

        self.bias = None
        if bias:

            self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)


    def build_filter(self) -> torch.Tensor:


        _filter = None
        _bias = None

        n_out, n_in, n_r, n_k, n_k = self.weight.shape
        self.register_buffer("_filter", torch.empty((n_out, 4, n_in, n_r, n_k, n_k)), persistent=False)
        for r in range(4):
            self._filter[:, r, :, :, :, :] = rotate_p4(self.weight, r)

        if self.bias is not None:
            self.register_buffer("_bias", self.bias.repeat(4, 1).T, persistent=False)

        if self.bias is not None:
            for i in range(4):
                self._bias[:, i] = self.bias

        else:
            self._bias = None

        device = self.weight.device
        self._filter = self._filter.to(device)
        if self.bias is not None:
            self._bias = self._bias.to(device)

        return self._filter, self._bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        if _bias is not None:
            assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size)

        _filter = _filter.reshape(self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size)
        if _bias is not None:
            _bias = _bias.reshape(self.out_channels * 4)

        x = x.view(x.shape[0], self.in_channels * 4, x.shape[-2], x.shape[-1])

        out = torch.conv2d(x, _filter, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=_bias)

        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])


class GroupBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super(GroupBatchNorm2d, self).__init__()

        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device)

    def forward(self, x):
        x_new = torch.empty_like(x)

        x_0 = self.bn(x[:, :, 0, :, :])
        for r in range(4):
            x_new[:, :, r, :, :] = x_0.rot90(r, dims=(-2, -1))

        return x_new


class C4CNN(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1):

        super(C4CNN, self).__init__()

        self.block1 = nn.Sequential(
            LiftingConv2d(n_channels, 24, kernel_size=7, padding=1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            GroupConv2d(24, 48, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )

        self.pool1 = nn.AvgPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

        self.block3 = nn.Sequential(
            GroupConv2d(48, 48, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            GroupConv2d(48, 96, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )

        self.pool2 = nn.AvgPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

        self.block5 = nn.Sequential(
            GroupConv2d(96, 96, kernel_size=5, padding=2, bias=False),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            GroupConv2d(96, 64, kernel_size=5, padding=1, bias=False),
            nn.ReLU()
        )

        self.pool3 = nn.AvgPool3d((1, 5, 5), (1, 1, 1), (0, 0, 0))

        self.gpool = torch.nn.MaxPool3d((4, 1, 1), (1, 1, 1), (0, 0, 0))

        self.fully_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_classes),
        )



    def forward(self, x: torch.Tensor, rot=None):

        first = self.block1(x)
        x = self.block2(first)
        x = self.pool1(x)

        x = self.block3(x)
        mid_feats = self.block4(x)
        x = self.pool2(mid_feats)

        x = self.block5(x)
        x = self.block6(x)

        x = self.pool3(x)
        x = self.gpool(x)

        x = self.fully_net(x.reshape(x.shape[0], -1))

        if rot is not None:
            idx = round(rot*4 / (2*np.pi)) % 4
        else:
            idx = 0

        return x

