"""Multi-timescale continuous encoder: fast/mid/slow branches with cross-talk
(tech_spec.md sections 6 and 9).
"""

from __future__ import annotations

import torch
from torch import nn

from aevum.models.dynamics.cell import ContinuousTimeCell
from aevum.utils.state import MultiTimescaleState


class MultiTimescaleDynamics(nn.Module):
    """Three ``ContinuousTimeCell`` branches (fast/mid/slow) with small cross-timescale projections.

    Generic continuous-time dynamics core shared by both the encoder (driven by
    frontend features) and the decoder (driven by received event embeddings) —
    tech_spec.md sections 6, 9, and 27.
    """

    def __init__(
        self,
        input_dim: int,
        fast_dim: int = 192,
        mid_dim: int = 192,
        slow_dim: int = 192,
        tau_ranges: dict[str, tuple[float, float]] | None = None,
        candidate_uses_hidden: bool = True,
    ) -> None:
        super().__init__()
        tau_ranges = tau_ranges or {
            "fast": (0.010, 0.080),
            "mid": (0.050, 0.500),
            "slow": (0.300, 5.000),
        }

        # Cross-timescale inputs are small projections of the *other* branch's
        # previous state, concatenated onto the frontend features before the cell.
        self.fast_cross = nn.Linear(mid_dim, mid_dim // 4)
        self.mid_cross = nn.Linear(fast_dim, fast_dim // 4)
        self.mid_cross_slow = nn.Linear(slow_dim, slow_dim // 4)
        self.slow_cross = nn.Linear(mid_dim, mid_dim // 4)

        # candidate_uses_hidden=False drops each cell's own h_{t-1} from its candidate's
        # input (cross-branch inputs, part of x_t here, are unaffected) -- see
        # docs/reports/stage1_v0.md Run 14 (encoder) and Run 27 (decoder) for why.
        self.fast_cell = ContinuousTimeCell(input_dim + mid_dim // 4, fast_dim, *tau_ranges["fast"], candidate_uses_hidden=candidate_uses_hidden)
        self.mid_cell = ContinuousTimeCell(input_dim + fast_dim // 4 + slow_dim // 4, mid_dim, *tau_ranges["mid"], candidate_uses_hidden=candidate_uses_hidden)
        self.slow_cell = ContinuousTimeCell(input_dim + mid_dim // 4, slow_dim, *tau_ranges["slow"], candidate_uses_hidden=candidate_uses_hidden)

    def forward(self, f_t: torch.Tensor, state: MultiTimescaleState, dt: float) -> MultiTimescaleState:
        """One 10 ms update of all three branches given frontend features ``f_t``."""
        fast_in = torch.cat([f_t, self.fast_cross(state.mid)], dim=-1)
        mid_in = torch.cat([f_t, self.mid_cross(state.fast), self.mid_cross_slow(state.slow)], dim=-1)
        slow_in = torch.cat([f_t, self.slow_cross(state.mid)], dim=-1)

        new_fast = self.fast_cell(fast_in, state.fast, dt)
        new_mid = self.mid_cell(mid_in, state.mid, dt)
        new_slow = self.slow_cell(slow_in, state.slow, dt)

        return MultiTimescaleState(fast=new_fast, mid=new_mid, slow=new_slow)
