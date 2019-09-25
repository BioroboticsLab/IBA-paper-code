import torch
import torch.nn as nn
import math


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels, device):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float, device=device)  # 1, 2, 3, 4
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)  # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, padding=0,
                              kernel_size=kernel_size, groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.to(device)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(self.pad(x))
