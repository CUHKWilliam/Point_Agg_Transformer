import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.utils.coords import get_coords_map

class MinkowskiEncoder(ME.MinkowskiNetwork):
    def __init__(self,
                dim,
                channels,
                kernel_size,
                stride,
                group=(4,),
                residual=True):
        super(MinkowskiEncoder, self).__init__(dim)
        self.conv = nn.ModuleList([])
        for i, (c, k, s) in enumerate(zip(channels, kernel_size, stride,)):
            conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=c,
                    out_channels=channels[i+1],
                    kernel_size=k,
                    stride=s,
                    dimension=dim),
                ME.MinkowskiBatchNorm(channels[i+1]),
                ME.MinkowskiReLU(),
            )
            self.conv.append(conv)
        self.dim = dim
        self.residual = residual

    def forward(self, x):
        """
        x: Hyper correlation. B L H_q W_q H_s W_s
        """
        residuals = []
        for conv in self.conv:
            if self.residual:
                residuals.append(x)
            x = conv(x)
        return x, residuals

class MinkowskiUpsample(ME.MinkowskiNetwork):
    def __init__(self,
                dim,
                channels,
                kernel_size,
                stride,
                residual=False):
        super(MinkowskiUpsample, self).__init__(dim)
        self.conv = nn.ModuleList([])
        for i, (c, k, s) in enumerate(zip(channels, kernel_size, stride,)):
            conv = nn.Sequential(
                ME.MinkowskiConvolutionTranspose(
                    in_channels=c,
                    out_channels=channels[i+1],
                    kernel_size=k,
                    stride=s,
                    dimension=dim),
                ME.MinkowskiBatchNorm(channels[i+1]),
                ME.MinkowskiReLU(),
            )
            self.conv.append(conv)
        self.dim = dim
        self.residual = residual

    def forward(self, x):
        """
        x: Hyper correlation. B L H_q W_q H_s W_s
        """
        residuals = []
        for conv in self.conv:
            if self.residual:
                residuals.append(x)
            x = conv(x)
        return x, residuals