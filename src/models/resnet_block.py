import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, stride: int = 1,
                 layer: nn.Module = nn.Conv3d, normalization: nn.Module = nn.BatchNorm3d,
                 activation: nn.Module = nn.ReLU) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.layer = layer
        self.normalization = normalization
        self.activation = activation

        # Full pre-activation block
        self.weight_block_0 = nn.Sequential(
            self.normalization(self.num_channels),
            self.activation(inplace=True),
            self.layer(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=self.stride,
                       padding=self.padding, bias=False)
        )

        self.weight_block_1 = nn.Sequential(
            self.normalization(self.num_channels),
            self.activation(inplace=True),
            self.layer(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=self.stride,
                       padding=self.padding, bias=False)
        )

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.weight_block_0(x)
        out = self.weight_block_1(out)

        out = identity + out

        return out
