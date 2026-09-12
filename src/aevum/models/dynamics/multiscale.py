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
        adaptive_tau: bool = True,
        tau_uses_hidden: bool = True,
        fixed_tau: float | None = None,
        disable_cross_connections: bool = False,
    ) -> None:
        super().__init__()
        tau_ranges = tau_ranges or {
            "fast": (0.010, 0.080),
            "mid": (0.050, 0.500),
            "slow": (0.300, 5.000),
        }
        self.disable_cross_connections = disable_cross_connections

        # Cross-timescale inputs are small projections of the *other* branch's
        # previous state, concatenated onto the frontend features before the cell.
        # disable_cross_connections=True removes this cross-talk entirely (tested
        # only on the decoder side so far -- docs/reports/stage1_v0.md Run 27).
        if disable_cross_connections:
            self.fast_cross = self.mid_cross = self.mid_cross_slow = self.slow_cross = None
            fast_in_dim = mid_in_dim = slow_in_dim = input_dim
        else:
            self.fast_cross = nn.Linear(mid_dim, mid_dim // 4)
            self.mid_cross = nn.Linear(fast_dim, fast_dim // 4)
            self.mid_cross_slow = nn.Linear(slow_dim, slow_dim // 4)
            self.slow_cross = nn.Linear(mid_dim, mid_dim // 4)
            fast_in_dim = input_dim + mid_dim // 4
            mid_in_dim = input_dim + fast_dim // 4 + slow_dim // 4
            slow_in_dim = input_dim + mid_dim // 4

        # candidate_uses_hidden=False drops each cell's own h_{t-1} from its candidate's
        # input (cross-branch inputs, part of x_t here, are unaffected) -- see
        # docs/reports/stage1_v0.md Run 14 (encoder) and Run 27 (decoder) for why.
        # adaptive_tau/tau_uses_hidden mirror the same options on ContinuousTimeCell
        # (Run 15's fixed-tau test). When adaptive_tau=False and fixed_tau is not
        # given explicitly, each branch defaults to its own tau_range midpoint
        # (matching Run 15's single-branch design) rather than one shared value,
        # since fast/mid/slow have deliberately different natural timescales.
        for name, (tau_min, tau_max) in tau_ranges.items():
            branch_fixed_tau = fixed_tau if fixed_tau is not None else (tau_min + tau_max) / 2
            cell = ContinuousTimeCell(
                {"fast": fast_in_dim, "mid": mid_in_dim, "slow": slow_in_dim}[name],
                {"fast": fast_dim, "mid": mid_dim, "slow": slow_dim}[name],
                tau_min,
                tau_max,
                candidate_uses_hidden=candidate_uses_hidden,
                adaptive_tau=adaptive_tau,
                tau_uses_hidden=tau_uses_hidden,
                fixed_tau=branch_fixed_tau if not adaptive_tau else None,
            )
            setattr(self, f"{name}_cell", cell)

    def forward(self, f_t: torch.Tensor, state: MultiTimescaleState, dt: float) -> MultiTimescaleState:
        """One 10 ms update of all three branches given frontend features ``f_t``."""
        if self.disable_cross_connections:
            fast_in, mid_in, slow_in = f_t, f_t, f_t
        else:
            fast_in = torch.cat([f_t, self.fast_cross(state.mid)], dim=-1)
            mid_in = torch.cat([f_t, self.mid_cross(state.fast), self.mid_cross_slow(state.slow)], dim=-1)
            slow_in = torch.cat([f_t, self.slow_cross(state.mid)], dim=-1)

        new_fast = self.fast_cell(fast_in, state.fast, dt)
        new_mid = self.mid_cell(mid_in, state.mid, dt)
        new_slow = self.slow_cell(slow_in, state.slow, dt)

        return MultiTimescaleState(fast=new_fast, mid=new_mid, slow=new_slow)
