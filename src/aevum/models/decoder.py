"""Continuous decoder: driven by received events, evolves even between them
(tech_spec.md sections 26-28, 43).

Stage 1 simplification: there is no event gate yet, so every step is "an
event" carrying the full latent ``z_t`` as ``u_t``. Sections 43/26 still apply
architecturally — the decoder has its own continuous-time state that keeps
evolving step to step, it just happens to receive input every step for now.

Architecture as validated end-to-end in docs/reports/stage1_v0.md (Runs
26-31): same "don't force a slow-decaying state to be the sole transport
channel" principle as the encoder (../../../lessons-continuous-time-
dynamics.md), applied here as a direct skip from the received event
straight to the output. The skip's gate is a *fixed* scalar, not a learned
parameter: a free/learnable gate was found to collapse toward suppressing
the temporal branch almost entirely (Run 26), which also brought back the
robotic-sounding quality that a properly-contributing temporal branch
fixes (Run 28). Each branch's own candidate self-recurrence is disabled by
default (Run 28: this — not the gate — was the actual source of
hidden-state saturation and large gradient spikes); adaptive tau and
cross-timescale connections are kept, since fixing tau was a regression
(Run 29) and disabling cross-connections was a wash (Run 30).
"""

from __future__ import annotations

import torch
from torch import nn

from aevum.models.dynamics.multiscale import MultiTimescaleDynamics
from aevum.utils.state import MultiTimescaleState


class ContinuousDecoder(nn.Module):
    def __init__(
        self,
        event_dim: int = 384,
        fast_dim: int = 192,
        mid_dim: int = 192,
        slow_dim: int = 192,
        output_dim: int = 384,
        step_seconds: float = 0.01,
        candidate_uses_hidden: bool = False,
        adaptive_tau: bool = True,
        tau_uses_hidden: bool = True,
        fixed_tau: float | None = None,
        disable_cross_connections: bool = False,
        skip_gate: float = 0.05,
    ) -> None:
        super().__init__()
        self.step_seconds = step_seconds
        self.dynamics = MultiTimescaleDynamics(
            event_dim,
            fast_dim,
            mid_dim,
            slow_dim,
            candidate_uses_hidden=candidate_uses_hidden,
            adaptive_tau=adaptive_tau,
            tau_uses_hidden=tau_uses_hidden,
            fixed_tau=fixed_tau,
            disable_cross_connections=disable_cross_connections,
        )

        fused_dim = fast_dim + mid_dim + slow_dim
        self.norm = nn.RMSNorm(fused_dim)
        self.to_output = nn.Linear(fused_dim, output_dim)

        # Direct skip from the received event to the output, identity-initialized
        # when dims match. Gate is a fixed buffer (not nn.Parameter) -- Run 26
        # found a learnable gate collapses toward zero instead of learning to
        # use the temporal branch.
        self.skip = nn.Linear(event_dim, output_dim)
        if event_dim == output_dim:
            with torch.no_grad():
                self.skip.weight.copy_(torch.eye(output_dim))
                self.skip.bias.zero_()
        self.register_buffer("skip_gate", torch.tensor(skip_gate))

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
        y_t = self.to_output(self.norm(fused)) + self.skip_gate * self.skip(u_t)
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
