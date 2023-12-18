import numpy as np

import torch

from torch import nn

import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Conv class"""

    def __init__(
        self, in_channels: int, out_channels: int, downsample: bool = True
    ) -> None:
        """Constructor for the double conv class

        Parameters:
        -----------
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
        """
        super(DoubleConv, self).__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the double conv layer

        Parameters:
        -----------
            x: torch.Tensor
                input tensor

        Returns:
        --------
            torch.Tensor:
                output tensor
        """
        x = self.conv(x)

        if self.downsample:
            x = F.max_pool2d(x, 2)

        return x


class UpSample(nn.Module):
    """UpSample class"""

    def __init__(
        self, in_channels: int, out_channels: int, use_conv_transpose: bool = False
    ) -> None:
        """Constructor for the UpSample class

        Parameters:
        -----------
            in_channels: int
                number of input channels
            out_channels: int
                number of output channels
            use_conv_transpose: bool
                whether to use transpose convolution or not
        """
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_transpose = use_conv_transpose

        if self.use_conv_transpose:
            self.tconv = nn.Sequential(
                nn.ConvTranspose2d(
                    self.in_channels, self.out_channels, kernel_size=2, stride=2
                ),
                nn.BatchNorm2d(self.out_channels),
                # nn.ELU(),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.Dropout2d(0.1),
                # nn.ELU(),
                nn.ReLU(),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DoubleConv(self.in_channels, self.out_channels, downsample=False),
            )

    def forward(self, x_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpSample class

        Parameters:
        -----------
            x_prev: torch.Tensor
                input tensor from the previous layer
            x: torch.Tensor
                input tensor from the skip connection

        Returns:
        --------
            torch.Tensor:
                output tensor
        """
        if self.use_conv_transpose:
            x_prev = self.tconv(x_prev)
        else:
            x_prev = self.up(x_prev)

        # pad x_prev to match x
        diff_h = x_prev.size()[2] - x.size()[2]
        diff_w = x_prev.size()[3] - x.size()[3]
        x = F.pad(
            x,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )
        return x_prev + x


class DepthNet(nn.Module):
    def __init__(
        self, in_channels, init_weights: bool = True, use_transpose_conv: bool = False
    ) -> None:
        super().__init__()
        # downsampling
        self.downsample1 = DoubleConv(in_channels, 64)
        self.downsample2 = DoubleConv(64, 128)
        self.downsample3 = DoubleConv(128, 256)
        self.downsample4 = DoubleConv(256, 512)
        # upsampling
        self.upsample1 = UpSample(512, 256, use_transpose_conv)
        self.upsample2 = UpSample(256, 128, use_transpose_conv)
        self.upsample3 = UpSample(128, 64, use_transpose_conv)
        self.upsample4 = UpSample(64, 64, use_transpose_conv)
        # 1x1 convolution
        self.conv = nn.Conv2d(64, 1, 1)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        """Iniitializes the weights of the network according to the Kaiming uniform distribution
        to avoid vanishing or exploding gradients by keeping the mean of the activations around 0 and the variance around 1
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # kaiming initialization
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsampling
        x1 = self.downsample1(x)
        del x
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)
        x4 = self.downsample4(x3)
        # upsampling
        x_prev = self.upsample1(x4, x3)
        del x4
        x_prev = self.upsample2(
            x_prev,
            x2,
        )
        del x2
        x_prev = self.upsample3(x_prev, x1)
        x_prev = self.upsample4(x_prev, x1)
        del x1
        # 1x1 convolution
        x_prev = (
            F.relu(self.conv(x_prev + 1e-3)) + 1e-6
        )  # add small value to avoid log(0)

        return x_prev
