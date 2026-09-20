"""Reconstruction losses: waveform L1 + mel L1 + multi-resolution STFT (tech_spec.md section 31)."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn


def align_for_reconstruction(x: torch.Tensor, x_hat: torch.Tensor, delay: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift ``x_hat`` against ``x`` by ``delay`` samples before comparing them.

    The frontend/decoder/generator stack is strictly causal: with left-only
    padding, output sample ``p`` (within a 240-sample frame starting at
    ``240*floor(p/240)``) only ever depends on input up to that frame's start,
    not on the samples within the frame it is nominally reconstructing. Naive
    same-index comparison (``x_hat[p]`` vs ``x[p]``) therefore asks the model
    to predict up to 239/240 of its own frame before having observed it. ``x``
    is delayed by exactly one frontend frame (``delay=240`` samples = 10 ms)
    relative to ``x_hat``: comparing ``x_hat[delay:]`` against ``x[:-delay]``
    means position ``p`` (post-shift) reconstructs the real input sample the
    model had actually already seen by the time it produced that output.

    This does not, by itself, add streaming latency to the model -- it only
    fixes what the loss compares against what. See docs/reports/stage1_v0.md.
    """
    if delay <= 0:
        return x, x_hat
    return x[..., :-delay], x_hat[..., delay:]


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of spectral-convergence + log-magnitude losses across several FFT sizes."""

    def __init__(self, fft_sizes: tuple[int, ...] = (256, 512, 1024, 2048), eps: float = 1e-2) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    @staticmethod
    def _magnitude(x: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, window=window, return_complex=True)
        return spec.abs()

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """``x``, ``x_hat``: ``[B, samples]``."""
        loss = x.new_zeros(())
        for n_fft in self.fft_sizes:
            mag = self._magnitude(x, n_fft)
            mag_hat = self._magnitude(x_hat, n_fft)

            spectral_convergence = torch.linalg.norm(mag - mag_hat, dim=(-2, -1)) / torch.linalg.norm(mag, dim=(-2, -1)).clamp_min(self.eps)
            log_mag = torch.nn.functional.l1_loss(torch.log(mag + self.eps), torch.log(mag_hat + self.eps))

            loss = loss + spectral_convergence.mean() + log_mag
        return loss / len(self.fft_sizes)


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_mels: int = 80,
        wav_weight: float = 1.0,
        mel_weight: float = 1.0,
        stft_weight: float = 1.0,
        eps: float = 1e-2,
        delay: int = 0,
    ) -> None:
        super().__init__()
        self.wav_weight = wav_weight
        self.mel_weight = mel_weight
        self.stft_weight = stft_weight
        self.eps = eps
        # See align_for_reconstruction: the model's output at sample p only depends on input up
        # to p's frame start, so naive same-index comparison scores it against unseen future
        # samples. delay=240 (one 10ms frontend frame) is the model's actual analysis lag.
        self.delay = delay

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=240, n_mels=n_mels
        )
        # Shared eps: both are instances of the same log(mag + eps) w.r.t. predicted
        # magnitude near zero -> 1/(mag_hat + eps) blow-up (docs/reports/stage1_v0.md, Run 6).
        self.stft_loss = MultiResolutionSTFTLoss(eps=eps)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``, ``x_hat``: ``[B, 1, samples]``."""
        x, x_hat = align_for_reconstruction(x, x_hat, self.delay)
        wav_loss = torch.nn.functional.l1_loss(x, x_hat)

        # log-compressed mel (power spectrogram is unbounded and would otherwise
        # swamp wav/stft in the weighted sum), matching common codec practice.
        mel_x = torch.log(self.mel(x.squeeze(1)) + self.eps)
        mel_x_hat = torch.log(self.mel(x_hat.squeeze(1)) + self.eps)
        mel_loss = torch.nn.functional.l1_loss(mel_x, mel_x_hat)

        stft_loss = self.stft_loss(x.squeeze(1), x_hat.squeeze(1))

        total = self.wav_weight * wav_loss + self.mel_weight * mel_loss + self.stft_weight * stft_loss
        return {"total": total, "wav": wav_loss, "mel": mel_loss, "stft": stft_loss}
