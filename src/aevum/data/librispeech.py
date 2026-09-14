"""LibriSpeech, resampled to the codec's target rate and chopped into fixed-length segments.

One epoch covers the whole split: every utterance is chopped into
non-overlapping ``segment_seconds``-long chunks, not sampled once at random.
The old scheme (one random crop per *file*, regardless of length) meant an
"epoch" only ever touched ~16% of a split's total audio (e.g. ~15.9h of
train-clean-100's 100.6h) -- see docs/reports/stage1_v0.md.

All progress reporting here uses plain ``print(..., flush=True)`` lines, not
``tqdm``: tqdm's in-place (carriage-return) redraw did not render at all
through ``!uv run ...`` in Colab, while plain flushed prints reliably did --
see docs/reports/stage1_v0.md.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

TARGET_SAMPLE_RATE = 24_000
LIBRISPEECH_SAMPLE_RATE = 16_000
_OPENSLR_BASE_URL = "http://www.openslr.org/resources/12/"
_PROGRESS_EVERY_BYTES = 200 * 1024 * 1024  # print every ~200MB downloaded
_PROGRESS_EVERY_FILES = 1000  # print every N files extracted/indexed


def _download_with_progress(url: str, dst: Path, hash_prefix: str | None) -> None:
    tmp = dst.with_name(dst.name + ".partial")
    sha256 = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_print = 0
        print(f"downloading {url} ({total / 1e9:.2f} GB)", flush=True)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            sha256.update(chunk)
            downloaded += len(chunk)
            if downloaded >= next_print:
                pct = downloaded / total * 100 if total else 0.0
                print(f"  downloaded {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
                next_print += _PROGRESS_EVERY_BYTES

    if hash_prefix and not sha256.hexdigest().startswith(hash_prefix):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch downloading {url}")
    tmp.rename(dst)
    print(f"download complete: {dst}", flush=True)


def _ensure_downloaded_and_extracted(root: Path, url: str, download: bool) -> None:
    """Fetch+extract a LibriSpeech split with visible progress (plain prints, see module docstring)."""
    if (root / "LibriSpeech" / url).is_dir():
        print(f"found existing '{url}' split under {root}, skipping download/extract", flush=True)
        return
    if not download:
        raise RuntimeError(f"Dataset split '{url}' not found under {root} and download=False.")

    from torchaudio.datasets.librispeech import _CHECKSUMS

    archive = root / f"{url}.tar.gz"
    download_url = _OPENSLR_BASE_URL + f"{url}.tar.gz"
    if not archive.is_file():
        _download_with_progress(download_url, archive, _CHECKSUMS.get(download_url))
    else:
        print(f"found existing archive {archive}, skipping download", flush=True)

    print(f"extracting {archive} ...", flush=True)
    with tarfile.open(archive, "r") as tar:
        members = tar.getmembers()
        total = len(members)
        print(f"  {total} files to extract", flush=True)
        for i, member in enumerate(members, 1):
            tar.extract(member, root)
            if i % _PROGRESS_EVERY_FILES == 0 or i == total:
                print(f"  extracted {i}/{total} files", flush=True)


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
        _ensure_downloaded_and_extracted(root, url, download)

        # torchaudio.datasets.LIBRISPEECH.__init__ does a blocking Path.glob() over
        # every audio file to build its file list, with no progress callback -- on a
        # Drive-mounted root each directory listing has real network latency, and
        # with ~28.5k files (train-clean-100) this can silently take a while.
        print(f"scanning '{url}' file list under {root} ...", flush=True)
        t0 = time.perf_counter()
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=str(root), url=url, download=False)
        print(f"found {len(self.dataset)} files in {time.perf_counter() - t0:.0f}s", flush=True)

        self.segment_samples = int(segment_seconds * TARGET_SAMPLE_RATE)
        self._native_segment_samples = int(segment_seconds * LIBRISPEECH_SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(LIBRISPEECH_SAMPLE_RATE, TARGET_SAMPLE_RATE)

        index_path = root / f"segment_index_{url}_{segment_seconds}s.json"
        if index_path.exists():
            print(f"loading cached segment index from {index_path}", flush=True)
            self._index: list[tuple[int, int]] = [tuple(pair) for pair in json.loads(index_path.read_text())]
        else:
            self._index = self._build_index()
            index_path.write_text(json.dumps(self._index))
        print(f"{len(self._index)} segments ready", flush=True)

    def _build_index(self) -> list[tuple[int, int]]:
        # torchaudio.info() reads only the file header (fast, no waveform decode).
        # self.dataset._archive/get_metadata are torchaudio-internal but this is
        # the only way to get per-file paths/durations without loading audio.
        # Parallelized with threads (I/O-bound, especially over a Drive-mounted
        # --data-root where each file access has real network latency).
        archive = Path(self.dataset._archive)
        total = len(self.dataset)
        print(f"indexing {total} files ...", flush=True)

        def _num_frames(utterance_idx: int) -> int:
            filepath, *_ = self.dataset.get_metadata(utterance_idx)
            return torchaudio.info(str(archive / filepath)).num_frames

        frame_counts: list[int] = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            for i, num_frames in enumerate(pool.map(_num_frames, range(total)), 1):
                frame_counts.append(num_frames)
                if i % _PROGRESS_EVERY_FILES == 0 or i == total:
                    print(f"  indexed {i}/{total} files", flush=True)

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
