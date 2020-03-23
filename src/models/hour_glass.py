import torch
import torch.nn as nn


class HourGlass(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        out = self.decoder(hidden)

        return out
