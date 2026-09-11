"""Explicit streaming state containers (tech_spec.md section 47).

State is passed explicitly rather than held as hidden mutable buffers inside
``nn.Module`` instances, so batching, resets, and streaming inference stay
simple and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiTimescaleState:
    """Hidden state of the fast/mid/slow continuous-time branches."""

    fast: torch.Tensor
    mid: torch.Tensor
    slow: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, fast_dim: int, mid_dim: int, slow_dim: int, device: torch.device) -> MultiTimescaleState:
        return cls(
            fast=torch.zeros(batch_size, fast_dim, device=device),
            mid=torch.zeros(batch_size, mid_dim, device=device),
            slow=torch.zeros(batch_size, slow_dim, device=device),
        )
