#!/usr/bin/env python3
"""Stage 1 training: dense continuous autoencoder, no quantization/predictor/event gate.

Validates the architectural hypothesis in isolation (tech_spec.md section 41,
Stage 1) before any codec-specific machinery is added on top.

Usage:
    uv run scripts/train_stage1.py --data-root data/raw --steps 20000
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AEVUM Stage 1 dense continuous autoencoder")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="train-clean-100")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = LibriSpeechSegments(
        root=args.data_root, url=args.librispeech_url, segment_seconds=args.segment_seconds
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = DenseContinuousAutoencoder().to(device)
    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.steps:
        for waveform in loader:
            waveform = waveform.to(device)

            reconstructed = model(waveform)
            losses = criterion(waveform, reconstructed)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            if step % args.log_every == 0:
                print(
                    f"step {step:>7d}  total {losses['total'].item():.4f}  "
                    f"wav {losses['wav'].item():.4f}  mel {losses['mel'].item():.4f}  "
                    f"stft {losses['stft'].item():.4f}"
                )

            if step % args.checkpoint_every == 0 and step > 0:
                torch.save(model.state_dict(), checkpoint_dir / f"stage1_step{step}.pt")

            step += 1
            if step >= args.steps:
                break

    torch.save(model.state_dict(), checkpoint_dir / "stage1_final.pt")


if __name__ == "__main__":
    main()
