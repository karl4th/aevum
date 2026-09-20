"""LibriSpeech, resampled to the codec's target rate and chopped into fixed-length segments.

One epoch covers the whole split: every utterance is chopped into
``segment_seconds``-long chunks (mostly non-overlapping, with one small
overlap at the tail of each utterance so the last few seconds aren't
permanently unseen -- see ``LibriSpeechSegments._build_index``), not sampled
once at random. The old scheme (one random crop per *file*, regardless of
length) meant an "epoch" only ever touched ~16% of a split's total audio
(e.g. ~15.9h of train-clean-100's 100.6h) -- see docs/reports/stage1_v0.md.

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
_PROGRESS_EVERY_BYTES = 10 * 1024 * 1024  # print every ~10MB downloaded
_PROGRESS_EVERY_FILES = 1000  # print every N files extracted/indexed
_SOCKET_TIMEOUT_SECONDS = 30  # a stalled connection raises instead of hanging silently forever
_DOWNLOAD_ATTEMPTS = 5


def _sha256_matches(path: Path, hash_prefix: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest().startswith(hash_prefix)


def _download_with_progress(url: str, dst: Path, hash_prefix: str | None) -> None:
    tmp = dst.with_name(dst.name + ".partial")

    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        sha256 = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as response, open(tmp, "wb") as f:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                next_print = 0
                print(f"downloading {url} ({total / 1e9:.2f} GB) [attempt {attempt}/{_DOWNLOAD_ATTEMPTS}]", flush=True)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha256.update(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_print:
                        pct = downloaded / total * 100 if total else 0.0
                        print(f"  downloaded {downloaded / 1e9:.3f}/{total / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
                        next_print += _PROGRESS_EVERY_BYTES
            break
        except (TimeoutError, OSError) as e:
            print(f"  download stalled/failed on attempt {attempt}/{_DOWNLOAD_ATTEMPTS}: {e!r}", flush=True)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            print("  retrying from scratch...", flush=True)
    else:
        raise RuntimeError(f"failed to download {url} after {_DOWNLOAD_ATTEMPTS} attempts")

    if hash_prefix and not sha256.hexdigest().startswith(hash_prefix):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch downloading {url}")
    tmp.rename(dst)
    print(f"download complete: {dst}", flush=True)


def _extraction_marker(root: Path, url: str) -> Path:
    """Written only after a full, verified extraction -- directory existence alone (the old
    check) is also true mid-extraction or after an interrupted extract, so a following run
    would silently treat a partial split as complete (docs/reports/stage1_v0.md)."""
    return root / f".extracted_{url}.json"


def _missing_or_incomplete_members(archive: Path, root: Path) -> list[tarfile.TarInfo]:
    """Regular-file members from ``archive`` that aren't already present on disk with the
    right size -- cheap (stat, not content-hash) verification that a directory left over
    from an interrupted extract is actually complete, without unconditionally re-extracting
    (which can be slow and, per a permission error hit while testing this fix, is not always
    safe to just blindly overwrite on every run)."""
    missing = []
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = root / member.name
            if not target.is_file() or target.stat().st_size != member.size:
                missing.append(member)
    return missing


def _ensure_downloaded_and_extracted(root: Path, url: str, download: bool) -> None:
    """Fetch+extract a LibriSpeech split with visible progress (plain prints, see module docstring)."""
    marker = _extraction_marker(root, url)
    if marker.exists():
        print(f"found existing '{url}' split under {root} (extraction verified complete), skipping", flush=True)
        return

    from torchaudio.datasets.librispeech import _CHECKSUMS

    archive = root / f"{url}.tar.gz"
    download_url = _OPENSLR_BASE_URL + f"{url}.tar.gz"
    expected_hash = _CHECKSUMS.get(download_url)
    split_dir = root / "LibriSpeech" / url

    if split_dir.is_dir() and not archive.is_file() and download:
        # Get the archive so the directory can actually be verified below, instead of
        # trusting it unverified just because download=True happens to be set.
        print(f"found '{url}' directory but no archive/marker -- downloading archive to verify completeness ...", flush=True)
        _download_with_progress(download_url, archive, expected_hash)

    if split_dir.is_dir() and archive.is_file():
        print(f"found '{url}' directory and archive but no completion marker -- verifying against the archive ...", flush=True)
        missing = _missing_or_incomplete_members(archive, root)
        if not missing:
            print(f"verified complete: all files from {archive} present, writing completion marker", flush=True)
            marker.write_text(json.dumps({"url": url, "archive_sha256_prefix": expected_hash}))
            return
        print(f"  {len(missing)} file(s) missing/incomplete (interrupted previous extract) -- re-extracting those", flush=True)
    elif split_dir.is_dir() and not archive.is_file():
        # download=False and no archive to verify against, so completeness genuinely can't be
        # checked. Trust the existing directory for *this* run, but deliberately do not write a
        # completion marker: writing one here would make the *next* run's bare marker.exists()
        # check treat this same unverified directory as verified, which is the exact bug this
        # marker scheme exists to prevent. Every run re-does this (cheap) check until a real
        # verification (archive present, or a fresh extract) can write the marker for real.
        print(
            f"found '{url}' directory under {root} but no archive to verify against (download=False) -- "
            "trusting it for this run only (NOT writing a completion marker); pass download=True or "
            "supply the archive to get a verified, cached extract",
            flush=True,
        )
        return
    else:
        missing = None  # fresh extract, not a partial-recovery re-verify

    if not archive.is_file():
        if not download:
            raise RuntimeError(f"Dataset split '{url}' not found under {root} and download=False.")
        _download_with_progress(download_url, archive, expected_hash)
    elif expected_hash and not _sha256_matches(archive, expected_hash):
        if not download:
            raise RuntimeError(f"Archive {archive} failed checksum and download=False.")
        print(f"existing archive {archive} failed checksum, re-downloading", flush=True)
        archive.unlink()
        _download_with_progress(download_url, archive, expected_hash)

    print(f"extracting {archive} ...", flush=True)
    with tarfile.open(archive, "r") as tar:
        members = missing if missing is not None else tar.getmembers()
        total = len(members)
        print(f"  {total} files to extract", flush=True)
        for i, member in enumerate(members, 1):
            tar.extract(member, root)
            if i % _PROGRESS_EVERY_FILES == 0 or i == total:
                print(f"  extracted {i}/{total} files", flush=True)

    still_missing = _missing_or_incomplete_members(archive, root)
    if still_missing:
        raise RuntimeError(
            f"extraction of {archive} finished but {len(still_missing)} file(s) are still missing/"
            f"incomplete under {root} (e.g. {still_missing[0].name})"
        )
    marker.write_text(json.dumps({"url": url, "archive_sha256_prefix": expected_hash}))


_INDEX_SCHEMA_VERSION = 2


class LibriSpeechSegments(Dataset):
    """Fixed-length mono 24 kHz speech segments covering the full LibriSpeech split.

    Every utterance is chopped into ``segment_seconds``-long chunks (utterances
    shorter than one segment are kept as a single zero-padded chunk). A
    remainder shorter than one full segment at the end of a longer utterance
    is still covered by an extra segment that overlaps the previous one just
    enough to reach the end of the audio, rather than being dropped -- with
    non-overlapping-only chunking every epoch permanently never sees those
    tail samples (docs/reports/stage1_v0.md). The flat segment index is built
    once via ``torchaudio.info`` (header-only reads, no full decode) and
    cached to ``<root>/segment_index_<url>_<segment_seconds>s.json`` so it
    isn't rebuilt on every run; the cache is keyed to a fingerprint of the
    file list so a corpus that changed (e.g. re-extracted after an
    interrupted first attempt) invalidates it automatically.
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

        fingerprint = self._corpus_fingerprint()
        index_path = root / f"segment_index_{url}_{segment_seconds}s.json"
        cached = None
        if index_path.exists():
            try:
                loaded = json.loads(index_path.read_text())
            except json.JSONDecodeError:
                loaded = None
            # Pre-schema caches were a bare JSON array (list), not the current
            # {"schema_version": ..., "segments": [...]} dict -- .get() on a list raises
            # AttributeError instead of falling through to "stale, rebuild".
            cached = loaded if isinstance(loaded, dict) else None
        if (
            cached is not None
            and cached.get("schema_version") == _INDEX_SCHEMA_VERSION
            and cached.get("url") == url
            and cached.get("segment_seconds") == segment_seconds
            and cached.get("corpus_fingerprint") == fingerprint
        ):
            print(f"loading cached segment index from {index_path}", flush=True)
            self._index: list[tuple[int, int]] = [tuple(pair) for pair in cached["segments"]]
        else:
            if cached is not None:
                print(f"cached segment index at {index_path} is stale (corpus/schema changed), rebuilding", flush=True)
            self._index = self._build_index()
            payload = {
                "schema_version": _INDEX_SCHEMA_VERSION,
                "url": url,
                "segment_seconds": segment_seconds,
                "corpus_fingerprint": fingerprint,
                "segments": self._index,
            }
            tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload))
            tmp_path.replace(index_path)  # atomic on POSIX and Windows; no half-written cache on crash
        print(f"{len(self._index)} segments ready", flush=True)

    def _corpus_fingerprint(self) -> str:
        """Cheap (stat-only, no decode) hash of the file list this index was built from,
        so a corpus that changed on disk (partial re-extract, added/removed files) doesn't
        silently reuse an index built for a different set of files/offsets."""
        archive = Path(self.dataset._archive)
        total = len(self.dataset)

        def _stat(utterance_idx: int) -> tuple[str, int]:
            filepath, *_ = self.dataset.get_metadata(utterance_idx)
            return filepath, (archive / filepath).stat().st_size

        with ThreadPoolExecutor(max_workers=32) as pool:
            entries = sorted(pool.map(_stat, range(total)))

        digest = hashlib.sha256()
        for filepath, size in entries:
            digest.update(f"{filepath}:{size}\n".encode())
        return digest.hexdigest()

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
            if num_frames <= self._native_segment_samples:
                index.append((utterance_idx, 0))
                continue
            n_full_segments = num_frames // self._native_segment_samples
            offsets = [seg * self._native_segment_samples for seg in range(n_full_segments)]
            remainder = num_frames - n_full_segments * self._native_segment_samples
            if remainder > 0:
                # Overlaps the previous segment instead of dropping the tail: starts
                # `native_segment_samples - remainder` earlier so it still ends exactly
                # at num_frames. Every sample in the utterance is covered by at least
                # one segment in every epoch.
                offsets.append(num_frames - self._native_segment_samples)
            index.extend((utterance_idx, offset) for offset in offsets)
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
