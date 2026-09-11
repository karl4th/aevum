#!/usr/bin/env python3
"""Stage 1 training: dense continuous autoencoder, no quantization/predictor/event gate.

Validates the architectural hypothesis in isolation (tech_spec.md section 41,
Stage 1) before any codec-specific machinery is added on top.

Usage:
    uv run scripts/train_stage1.py --data-root data/raw --librispeech-url dev-clean --steps 5000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AEVUM Stage 1 dense continuous autoencoder")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=500)
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

    # Fixed held-out clip for periodic listening/eval — reconstruction quality on
    # varying training batches is too noisy step-to-step to judge progress by ear.
    torch.manual_seed(0)
    val_waveform = dataset[0].unsqueeze(0).to(device)

    model = DenseContinuousAutoencoder().to(device)
    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(samples_dir / "val_target.wav"), val_waveform[0].cpu(), 24_000)

    best_val_loss = float("inf")
    step = 0
    while step < args.steps:
        for waveform in loader:
            waveform = waveform.to(device)

            reconstructed = model(waveform)
            losses = criterion(waveform, reconstructed)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            if step % args.log_every == 0:
                print(
                    f"step {step:>7d}  total {losses['total'].item():.4f}  "
                    f"wav {losses['wav'].item():.4f}  mel {losses['mel'].item():.4f}  "
                    f"stft {losses['stft'].item():.4f}"
                )

            if step % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    val_recon = model(val_waveform)
                    val_loss = criterion(val_waveform, val_recon)["total"].item()
                model.train()

                torchaudio.save(str(samples_dir / f"val_step{step}.wav"), val_recon[0].cpu(), 24_000)
                print(f"  [val] step {step:>7d}  loss {val_loss:.4f}  (best {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torchaudio.save(str(samples_dir / "val_best.wav"), val_recon[0].cpu(), 24_000)
                    torch.save(model.state_dict(), checkpoint_dir / "stage1_best.pt")

            if step % args.checkpoint_every == 0 and step > 0:
                torch.save(model.state_dict(), checkpoint_dir / f"stage1_step{step}.pt")

            step += 1
            if step >= args.steps:
                break

    torch.save(model.state_dict(), checkpoint_dir / "stage1_final.pt")


if __name__ == "__main__":
    main()
