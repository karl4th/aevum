"""Continuous decoder: driven by received events, evolves even between them
(tech_spec.md sections 26-28, 43).

Stage 1 simplification: there is no event gate yet, so every step is "an
event" carrying the full latent ``z_t`` as ``u_t``. Sections 43/26 still apply
architecturally — the decoder has its own continuous-time state that keeps
evolving step to step, it just happens to receive input every step for now.
"""

from __future__ import annotations

import torch
from torch import nn

from aevum.models.dynamics.multiscale import MultiTimescaleDynamics
from aevum.utils.state import MultiTimescaleState


class ContinuousDecoder(nn.Module):
    def __init__(
        self,
        event_dim: int = 512,
        fast_dim: int = 192,
        mid_dim: int = 192,
        slow_dim: int = 192,
        output_dim: int = 384,
        step_seconds: float = 0.01,
        candidate_uses_hidden: bool = True,
    ) -> None:
        super().__init__()
        self.step_seconds = step_seconds
        self.dynamics = MultiTimescaleDynamics(event_dim, fast_dim, mid_dim, slow_dim, candidate_uses_hidden=candidate_uses_hidden)

        fused_dim = fast_dim + mid_dim + slow_dim
        self.norm = nn.RMSNorm(fused_dim)
        self.to_output = nn.Linear(fused_dim, output_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> MultiTimescaleState:
        return MultiTimescaleState.zeros(
            batch_size,
            self.dynamics.fast_cell.hidden_dim,
            self.dynamics.mid_cell.hidden_dim,
            self.dynamics.slow_cell.hidden_dim,
            device,
        )

    def step(self, u_t: torch.Tensor, state: MultiTimescaleState) -> tuple[torch.Tensor, MultiTimescaleState]:
        """One 10 ms update. ``u_t``: ``[B, event_dim]`` (zeros if no event) -> ``y_t``: ``[B, output_dim]``."""
        new_state = self.dynamics(u_t, state, self.step_seconds)
        fused = torch.cat([new_state.fast, new_state.mid, new_state.slow], dim=-1)
        y_t = self.to_output(self.norm(fused))
        return y_t, new_state

    def forward(self, u: torch.Tensor, state: MultiTimescaleState | None = None) -> tuple[torch.Tensor, MultiTimescaleState]:
        """``[B, T, event_dim]`` -> ``[B, T, output_dim]`` 100 Hz acoustic representation."""
        batch_size = u.shape[0]
        if state is None:
            state = self.initial_state(batch_size, u.device)

        y_steps = []
        for t in range(u.shape[1]):
            y_t, state = self.step(u[:, t, :], state)
            y_steps.append(y_t)
        return torch.stack(y_steps, dim=1), state
