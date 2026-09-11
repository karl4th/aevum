"""Causal waveform generator: 100 Hz acoustic representation -> 24 kHz PCM
(tech_spec.md section 29).

Uses nearest-neighbor upsampling + causal conv rather than transposed
convolutions, since it is simpler to stream correctly.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from aevum.models.frontend import CausalConv1d


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, stride=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.act(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor, mode="nearest")
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size=2 * factor + 1, stride=1)
        self.act = nn.SiLU()
        self.residual = CausalResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(self.upsample(x)))
        return self.residual(x)


class CausalWaveformGenerator(nn.Module):
    """Upsamples ``[B, input_dim, T_100Hz]`` -> ``[B, 1, T_100Hz * 240]`` (24 kHz PCM)."""

    _FACTORS: ClassVar[list[int]] = [2, 2, 3, 4, 5]  # total stride 240
    _CHANNELS: ClassVar[list[int]] = [256, 192, 128, 96, 64]

    def __init__(self, input_dim: int = 384) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, self._CHANNELS[0], kernel_size=1)

        blocks = []
        channels = self._CHANNELS + [self._CHANNELS[-1]]
        for i, factor in enumerate(self._FACTORS):
            blocks.append(UpsampleBlock(channels[i], channels[i + 1], factor))
        self.blocks = nn.ModuleList(blocks)

        self.to_wave = CausalConv1d(channels[-1], 1, kernel_size=7, stride=1)

    @property
    def total_stride(self) -> int:
        stride = 1
        for f in self._FACTORS:
            stride *= f
        return stride

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(y)
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.to_wave(x))
