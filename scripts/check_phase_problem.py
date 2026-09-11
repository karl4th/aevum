#!/usr/bin/env python3
"""Does matching magnitude alone (with no true phase) already sound noisy?

Context (docs/reports/stage1_v0.md, Run 8): after fixing a confirmed
gradient-explosion bug (eps too small in ReconstructionLoss), the Stage 1
real-clip overfit test still produces noisy, unintelligible audio despite a
healthy, monotonically-improving loss curve. `wav` (L1 on the raw waveform)
barely moved the entire run while `mel`/`stft` (magnitude-only losses)
dropped steadily -- consistent with the model satisfying magnitude
constraints while getting phase/fine time structure wrong.

This script isolates that question entirely from the model: take the
*target*'s own true magnitude spectrogram, throw away its true phase, and
reconstruct a waveform from magnitude alone via Griffin-Lim (iterative phase
estimation). If this ALSO sounds noisy/metallic, that's direct evidence the
noise problem is inherent to a magnitude-only reconstruction objective, not
a symptom of anything wrong with our specific model or training run.

Usage:
    uv run scripts/check_phase_problem.py
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from aevum.data.librispeech import LibriSpeechSegments

SAMPLE_RATE = 24_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Griffin-Lim magnitude-only reconstruction check")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=240)
    parser.add_argument("--n-iter", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="outputs/phase_check")
    args = parser.parse_args()

    dataset = LibriSpeechSegments(
        root=args.data_root, url=args.librispeech_url, segment_seconds=args.seconds, download=True
    )
    torch.manual_seed(0)
    target = dataset[args.index]  # [1, samples] -- same fixed clip used everywhere else in this report

    window = torch.hann_window(args.n_fft)
    spec = torch.stft(
        target,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.n_fft,
        window=window,
        return_complex=True,
    )
    magnitude = spec.abs()  # true phase discarded from here on

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.n_fft, power=1.0, n_iter=args.n_iter
    )
    reconstructed = griffin_lim(magnitude)
    if reconstructed.dim() == 1:
        reconstructed = reconstructed.unsqueeze(0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target, SAMPLE_RATE)
    torchaudio.save(str(out_dir / "griffin_lim_magnitude_only.wav"), reconstructed, SAMPLE_RATE)

    print(f"target samples: {target.shape[-1]}, griffin-lim samples: {reconstructed.shape[-1]}")
    print(f"n_fft={args.n_fft} hop_length={args.hop_length} n_iter={args.n_iter}")
    print(f"samples written to: {out_dir}")
    print("Listen to griffin_lim_magnitude_only.wav vs target.wav.")


if __name__ == "__main__":
    main()
