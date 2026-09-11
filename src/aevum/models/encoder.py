"""Continuous encoder: PCM -> fused multi-timescale latent z(t) (tech_spec.md sections 5, 6, 10).

Stage 1 of the training curriculum (dense continuous autoencoder, section 41):
every step is treated as if it were an event, so there is no quantization,
predictor, or event gate yet — this module only validates that the causal
frontend + continuous-time dynamics can carry enough information to
reconstruct speech.
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
        latent_dim: int = 512,
        step_seconds: float = 0.01,
    ) -> None:
        super().__init__()
        self.step_seconds = step_seconds
        self.frontend = CausalAcousticFrontend()
        self.dynamics = MultiTimescaleDynamics(self.frontend.output_dim, fast_dim, mid_dim, slow_dim)

        fused_dim = fast_dim + mid_dim + slow_dim
        self.norm = nn.RMSNorm(fused_dim)
        self.to_latent = nn.Linear(fused_dim, latent_dim)

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
        fused = torch.cat([new_state.fast, new_state.mid, new_state.slow], dim=-1)
        z_t = self.to_latent(self.norm(fused))
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
