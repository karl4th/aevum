"""Continuous-time cell with an adaptive time constant (tech_spec.md section 7).

This is the core recurrence used by every timescale branch of the encoder and
decoder: a leaky-integrator update where the decay ``alpha`` is predicted
from the current input and previous state, instead of being fixed.
"""

from __future__ import annotations

import torch
from torch import nn


class ContinuousTimeCell(nn.Module):
    """One adaptive leaky-integrator step: ``h_t = alpha_t * h_{t-1} + (1 - alpha_t) * u_t``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau_min: float = 0.005,
        tau_max: float = 5.0,
        candidate_uses_hidden: bool = True,
        adaptive_tau: bool = True,
        tau_uses_hidden: bool = True,
        fixed_tau: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.candidate_uses_hidden = candidate_uses_hidden
        self.adaptive_tau = adaptive_tau
        self.tau_uses_hidden = tau_uses_hidden
        self.fixed_tau = fixed_tau

        # candidate_uses_hidden=False drops h_{t-1} from the candidate's own input,
        # so old state re-enters the update only through the alpha-weighted memory
        # term, not a second time through a learned nonlinear recurrent transform
        # (docs/reports/stage1_v0.md, Run 13: that double path is the suspected
        # source of the mid/slow branch gradient explosions).
        candidate_input_dim = input_dim + hidden_dim if candidate_uses_hidden else input_dim
        self.candidate = nn.Linear(candidate_input_dim, hidden_dim)

        # adaptive_tau=False replaces the learned tau network with a constant, so
        # h_{t-1} can no longer influence its own decay rate at all (the
        # h_{t-1} -> tau_t -> alpha_t -> h_t feedback path suspected in Run 14).
        # tau_uses_hidden=False keeps tau adaptive but input-only (still learned,
        # still varies per step, just not state-dependent).
        if adaptive_tau:
            tau_input_dim = input_dim + hidden_dim if tau_uses_hidden else input_dim
            self.time_constant = nn.Linear(tau_input_dim, hidden_dim)
        else:
            if fixed_tau is None:
                raise ValueError("fixed_tau must be set when adaptive_tau=False")
            self.time_constant = None

        self._recording = False
        self._tau_log: list[torch.Tensor] = []
        self._alpha_log: list[torch.Tensor] = []
        self._candidate_log: list[torch.Tensor] = []

    def start_recording(self) -> None:
        """Begin accumulating per-step tau/alpha/candidate-pre-tanh values (for diagnostics only)."""
        self._recording = True
        self._tau_log = []
        self._alpha_log = []
        self._candidate_log = []

    def stop_recording(self) -> dict[str, torch.Tensor]:
        """Stop accumulating and return the flattened tau/alpha/candidate-pre-tanh values seen since ``start_recording``."""
        self._recording = False
        tau = torch.cat(self._tau_log) if self._tau_log else torch.empty(0)
        alpha = torch.cat(self._alpha_log) if self._alpha_log else torch.empty(0)
        candidate_pretanh = torch.cat(self._candidate_log) if self._candidate_log else torch.empty(0)
        self._tau_log = []
        self._alpha_log = []
        self._candidate_log = []
        return {"tau": tau, "alpha": alpha, "candidate_pretanh": candidate_pretanh}

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one observation step of size ``dt`` seconds.

        Args:
            x_t: ``[B, input_dim]`` input at this step.
            h_prev: ``[B, hidden_dim]`` previous hidden state.
            dt: observation interval in seconds (e.g. 0.01 for 10 ms).

        Returns:
            ``[B, hidden_dim]`` updated hidden state.
        """
        xh = torch.cat([x_t, h_prev], dim=-1)
        candidate_logit = self.candidate(xh if self.candidate_uses_hidden else x_t)
        u_t = torch.tanh(candidate_logit)

        if self._recording:
            self._candidate_log.append(candidate_logit.detach().flatten())

        if self.adaptive_tau:
            tau_input = xh if self.tau_uses_hidden else x_t
            tau_t = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.time_constant(tau_input))
        else:
            tau_t = torch.full_like(u_t, self.fixed_tau)

        alpha_t = torch.exp(-dt / tau_t)

        if self._recording:
            self._tau_log.append(tau_t.detach().flatten())
            self._alpha_log.append(alpha_t.detach().flatten())

        return alpha_t * h_prev + (1 - alpha_t) * u_t
