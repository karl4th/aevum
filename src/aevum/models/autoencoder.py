"""Stage 1 dense continuous autoencoder (tech_spec.md section 41, Stage 1).

Wires frontend -> multi-timescale encoder -> continuous decoder -> waveform
generator with no quantization, predictor, or event gate: every 10 ms step
carries the full latent ``z_t`` straight to the decoder. This is the
architectural sanity check that has to pass before anything else in the spec
is worth building.
"""

from __future__ import annotations

import torch
from torch import nn

from aevum.models.decoder import ContinuousDecoder
from aevum.models.encoder import ContinuousEncoder
from aevum.models.generator import CausalWaveformGenerator


class DenseContinuousAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 512, decoder_output_dim: int = 384) -> None:
        super().__init__()
        self.encoder = ContinuousEncoder(latent_dim=latent_dim)
        self.decoder = ContinuousDecoder(event_dim=latent_dim, output_dim=decoder_output_dim)
        self.generator = CausalWaveformGenerator(input_dim=decoder_output_dim)

    @property
    def total_stride(self) -> int:
        return self.encoder.frontend.total_stride

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, 1, samples]`` -> reconstructed ``[B, 1, samples']``."""
        z, _ = self.encoder(waveform)
        y, _ = self.decoder(z)
        return self.generator(y.transpose(1, 2))
