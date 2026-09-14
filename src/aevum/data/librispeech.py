"""LibriSpeech, resampled to the codec's target rate and chopped into fixed-length segments.

One epoch covers the whole split: every utterance is chopped into
non-overlapping ``segment_seconds``-long chunks, not sampled once at random.
The old scheme (one random crop per *file*, regardless of length) meant an
"epoch" only ever touched ~16% of a split's total audio (e.g. ~15.9h of
train-clean-100's 100.6h) -- see docs/reports/stage1_v0.md.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

TARGET_SAMPLE_RATE = 24_000
LIBRISPEECH_SAMPLE_RATE = 16_000


class LibriSpeechSegments(Dataset):
    """Fixed-length mono 24 kHz speech segments covering the full LibriSpeech split.

    Every utterance is chopped into non-overlapping ``segment_seconds``-long
    chunks (utterances shorter than one segment are kept as a single
    zero-padded chunk; a shorter tail chunk at the end of a longer utterance
    is dropped). The flat segment index is built once via ``torchaudio.info``
    (header-only reads, no full decode) and cached to
    ``<root>/segment_index_<url>_<segment_seconds>s.json`` so it isn't
    rebuilt on every run.
    """

    def __init__(
        self,
        root: str | Path = "data/raw",
        url: str = "train-clean-100",
        segment_seconds: float = 2.0,
        download: bool = True,
    ) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=str(root), url=url, download=download)
        self.segment_samples = int(segment_seconds * TARGET_SAMPLE_RATE)
        self._native_segment_samples = int(segment_seconds * LIBRISPEECH_SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(LIBRISPEECH_SAMPLE_RATE, TARGET_SAMPLE_RATE)

        index_path = root / f"segment_index_{url}_{segment_seconds}s.json"
        if index_path.exists():
            self._index: list[tuple[int, int]] = [tuple(pair) for pair in json.loads(index_path.read_text())]
        else:
            self._index = self._build_index()
            index_path.write_text(json.dumps(self._index))

    def _build_index(self) -> list[tuple[int, int]]:
        # torchaudio.info() reads only the file header (fast, no waveform decode).
        # self.dataset._archive/get_metadata are torchaudio-internal but this is
        # the only way to get per-file paths/durations without loading audio.
        # Parallelized with threads (I/O-bound, especially over a Drive-mounted
        # --data-root where each file access has real network latency) and
        # shown with a progress bar -- without both, this step looks identical
        # to a hang for the ~1-2 minutes (local disk) to much longer (network
        # mount) it can take on a fresh --data-root.
        archive = Path(self.dataset._archive)

        def _num_frames(utterance_idx: int) -> int:
            filepath, *_ = self.dataset.get_metadata(utterance_idx)
            return torchaudio.info(str(archive / filepath)).num_frames

        with ThreadPoolExecutor(max_workers=32) as pool:
            frame_counts = list(
                tqdm(
                    pool.map(_num_frames, range(len(self.dataset))),
                    total=len(self.dataset),
                    desc="indexing LibriSpeech segments",
                )
            )

        index: list[tuple[int, int]] = []
        for utterance_idx, num_frames in enumerate(frame_counts):
            n_segments = max(1, num_frames // self._native_segment_samples)
            index.extend((utterance_idx, seg * self._native_segment_samples) for seg in range(n_segments))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> torch.Tensor:
        utterance_idx, start = self._index[index]
        filepath, sample_rate, *_ = self.dataset.get_metadata(utterance_idx)
        if sample_rate != LIBRISPEECH_SAMPLE_RATE:
            raise ValueError(f"expected {LIBRISPEECH_SAMPLE_RATE} Hz source audio, got {sample_rate}")

        archive = Path(self.dataset._archive)
        waveform, _ = torchaudio.load(
            str(archive / filepath), frame_offset=start, num_frames=self._native_segment_samples
        )
        waveform = self.resample(waveform)  # [1, samples] at 24 kHz

        if waveform.shape[-1] < self.segment_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[:, : self.segment_samples]

        return waveform
