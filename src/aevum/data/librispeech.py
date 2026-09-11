"""LibriSpeech, resampled to the codec's target rate and cropped/padded to fixed-length segments."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

TARGET_SAMPLE_RATE = 24_000
LIBRISPEECH_SAMPLE_RATE = 16_000


class LibriSpeechSegments(Dataset):
    """Fixed-length mono 24 kHz speech segments drawn from a LibriSpeech split."""

    def __init__(
        self,
        root: str | Path = "data/raw",
        url: str = "train-clean-100",
        segment_seconds: float = 2.0,
        download: bool = True,
    ) -> None:
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=str(root), url=url, download=download)
        self.segment_samples = int(segment_seconds * TARGET_SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(LIBRISPEECH_SAMPLE_RATE, TARGET_SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform, sample_rate, *_ = self.dataset[index]
        if sample_rate != LIBRISPEECH_SAMPLE_RATE:
            raise ValueError(f"expected {LIBRISPEECH_SAMPLE_RATE} Hz source audio, got {sample_rate}")

        waveform = self.resample(waveform)  # [1, samples] at 24 kHz
        num_samples = waveform.shape[-1]

        if num_samples >= self.segment_samples:
            start = torch.randint(0, num_samples - self.segment_samples + 1, (1,)).item()
            waveform = waveform[:, start : start + self.segment_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_samples - num_samples))

        return waveform
