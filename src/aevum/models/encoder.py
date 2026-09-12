"""Continuous encoder: PCM -> fused multi-timescale latent z(t) (tech_spec.md sections 5, 6, 10).

Stage 1 of the training curriculum (dense continuous autoencoder, section 41):
every step is treated as if it were an event, so there is no quantization,
predictor, or event gate yet — this module only validates that the causal
frontend + continuous-time dynamics can carry enough information to
reconstruct speech.

Architecture as validated end-to-end in docs/reports/stage1_v0.md (Runs
16-19, 22, 31), not tech_spec.md's original design: the fused multi-
timescale state is not the sole path from frontend features to the latent.
A slow-decaying continuous state cannot safely be the sole transport
channel for full-bandwidth signal (see ../../../lessons-continuous-time-
dynamics.md for the general lesson) — forcing that produced gradient
explosions and, decisively, robotic-sounding audio (Run 11-19). Instead,
frontend features reach the latent directly (identity-initialized when
dimensions allow), and each timescale branch contributes on top through a
small-init learnable gate, so training starts close to the already-proven
frontend-only baseline and the branches have to earn their contribution.
Branches also have no cross-timescale connections and no self-recurrence
in their own candidate (Run 14/19/22): memory already exists via
alpha_t*h_{t-1}, a second recurrent path was never shown to help.
"""

from __future__ import annotations

import torch
from torch import nn

from aevum.models.dynamics.multiscale import MultiTimescaleDynamics
from aevum.models.frontend import CausalAcousticFrontend
from aevum.utils.state import MultiTimescaleState


class ContinuousEncoder(nn.Module):
    def __init__(
        self,
        fast_dim: int = 192,
        mid_dim: int = 192,
        slow_dim: int = 192,
        latent_dim: int | None = None,
        step_seconds: float = 0.01,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.step_seconds = step_seconds
        self.frontend = CausalAcousticFrontend()
        input_dim = self.frontend.output_dim
        latent_dim = input_dim if latent_dim is None else latent_dim

        self.dynamics = MultiTimescaleDynamics(
            input_dim,
            fast_dim,
            mid_dim,
            slow_dim,
            candidate_uses_hidden=False,
            disable_cross_connections=True,
        )

        # Direct instantaneous path: identity-initialized when dims match, so
        # z_t starts out equal to the already-proven frontend-only baseline
        # (docs/reports/stage1_v0.md Run 12) before the gated branches below
        # learn to contribute anything.
        self.w_x = nn.Linear(input_dim, latent_dim)
        if latent_dim == input_dim:
            with torch.no_grad():
                self.w_x.weight.copy_(torch.eye(latent_dim))
                self.w_x.bias.zero_()

        self.fast_proj = nn.Linear(fast_dim, latent_dim)
        self.mid_proj = nn.Linear(mid_dim, latent_dim)
        self.slow_proj = nn.Linear(slow_dim, latent_dim)
        self.fast_gate = nn.Parameter(torch.tensor(gate_init))
        self.mid_gate = nn.Parameter(torch.tensor(gate_init))
        self.slow_gate = nn.Parameter(torch.tensor(gate_init))

    def initial_state(self, batch_size: int, device: torch.device) -> MultiTimescaleState:
        return MultiTimescaleState.zeros(
            batch_size,
            self.dynamics.fast_cell.hidden_dim,
            self.dynamics.mid_cell.hidden_dim,
            self.dynamics.slow_cell.hidden_dim,
            device,
        )

    def step(self, f_t: torch.Tensor, state: MultiTimescaleState) -> tuple[torch.Tensor, MultiTimescaleState]:
        """One 10 ms update. ``f_t``: ``[B, frontend.output_dim]`` -> latent ``z_t``: ``[B, latent_dim]``."""
        new_state = self.dynamics(f_t, state, self.step_seconds)
        z_t = (
            self.w_x(f_t)
            + self.fast_gate * self.fast_proj(new_state.fast)
            + self.mid_gate * self.mid_proj(new_state.mid)
            + self.slow_gate * self.slow_proj(new_state.slow)
        )
        return z_t, new_state

    def forward(self, waveform: torch.Tensor, state: MultiTimescaleState | None = None) -> tuple[torch.Tensor, MultiTimescaleState]:
        """``[B, 1, samples]`` -> latent sequence ``[B, T_100Hz, latent_dim]``."""
        batch_size = waveform.shape[0]
        if state is None:
            state = self.initial_state(batch_size, waveform.device)

        features = self.frontend(waveform).transpose(1, 2)  # [B, T, frontend.output_dim]

        z_steps = []
        for t in range(features.shape[1]):
            z_t, state = self.step(features[:, t, :], state)
            z_steps.append(z_t)
        return torch.stack(z_steps, dim=1), state
