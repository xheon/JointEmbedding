from typing import Tuple
import torch
import torch.nn as nn


class SeparationNet(nn.Module):
    def __init__(self, encoder: nn.Module, decoder_fg: nn.Module, decoder_bg: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder_fg = decoder_fg
        self.decoder_bg = decoder_bg

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)

        # Split latent space 
        hidden_shape = list(encoded.size())
        hidden_shape[1] //= 2

        half_dim = self.encoder.num_features[-1] // 2
        split = torch.split(encoded, half_dim, dim=1)
        hidden_fg = split[0].reshape(hidden_shape)
        hidden_bg = split[1].reshape(hidden_shape)

        return hidden_fg, hidden_bg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_fg, hidden_bg = self.encode(x)

        decoded_fg = self.decoder_fg(hidden_fg)
        decoded_bg = self.decoder_bg(hidden_bg)

        return decoded_fg, decoded_bg
