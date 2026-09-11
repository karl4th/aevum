"""Causal acoustic frontend: PCM -> 100 Hz feature vector (tech_spec.md section 5)."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding, so no future sample ever leaks in."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalAcousticFrontend(nn.Module):
    """5 causal conv layers, total stride 240 (24 kHz -> 100 Hz)."""

    _CHANNELS: ClassVar[list[int]] = [1, 64, 128, 192, 256, 384]
    _STRIDES: ClassVar[list[int]] = [5, 4, 3, 2, 2]
    _KERNEL_MULT = 2  # kernel_size = stride * _KERNEL_MULT

    def __init__(self) -> None:
        super().__init__()
        layers = []
        for i, stride in enumerate(self._STRIDES):
            in_ch, out_ch = self._CHANNELS[i], self._CHANNELS[i + 1]
            kernel_size = max(stride * self._KERNEL_MULT, stride)
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, stride))
            layers.append(nn.SiLU())
        self.layers = nn.ModuleList(layers)

    @property
    def output_dim(self) -> int:
        return self._CHANNELS[-1]

    @property
    def total_stride(self) -> int:
        stride = 1
        for s in self._STRIDES:
            stride *= s
        return stride

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, 1, samples]`` -> ``[B, output_dim, T_100Hz]``."""
        x = waveform
        for layer in self.layers:
            x = layer(x)
        return x
