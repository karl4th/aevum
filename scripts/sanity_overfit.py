#!/usr/bin/env python3
"""Sanity check: can Stage 1 overfit a single fixed clip?

No dataset download required. Catches broken gradients / dead cells /
non-learning generator cheaply, and reports throughput (steps/sec) at the
given batch size and segment length to inform whether a full training run
is feasible in reasonable wall-clock time.

Usage:
    uv run scripts/sanity_overfit.py
"""

import argparse
import time
from pathlib import Path

import torch
import torchaudio

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def make_synthetic_clip(seconds: float, batch_size: int, device: torch.device) -> torch.Tensor:
    """A slowly-modulated multi-tone signal: closer to speech-like quasi-periodic
    structure than white noise, so the causal convs aren't fighting an
    unlearnable target."""
    t = torch.arange(int(seconds * SAMPLE_RATE), device=device) / SAMPLE_RATE
    f0 = 140.0 + 40.0 * torch.sin(2 * torch.pi * 0.7 * t)  # wandering "pitch"
    envelope = 0.5 * (1 + torch.sin(2 * torch.pi * 1.3 * t))
    phase = 2 * torch.pi * torch.cumsum(f0, dim=0) / SAMPLE_RATE
    signal = envelope * (torch.sin(phase) + 0.3 * torch.sin(3 * phase) + 0.15 * torch.sin(5 * phase))
    signal = signal / signal.abs().max()
    return signal.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)


def load_real_clip(
    data_root: str, librispeech_url: str, index: int, seconds: float, batch_size: int, device: torch.device
) -> torch.Tensor:
    """One fixed real LibriSpeech utterance, repeated across the batch dim.

    Uses the same dataset/index convention as train_stage1.py's held-out
    validation clip (index 0), so results are directly comparable.
    """
    dataset = LibriSpeechSegments(root=data_root, url=librispeech_url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples], fixed crop given the seed above
    return clip.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit sanity check for the Stage 1 autoencoder")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--real-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    model = DenseContinuousAutoencoder().to(device)
    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.source == "real":
        waveform = load_real_clip(args.data_root, args.librispeech_url, args.real_index, args.seconds, args.batch_size, device)
    else:
        waveform = make_synthetic_clip(args.seconds, args.batch_size, device)
    num_frames = waveform.shape[-1] // model.total_stride
    waveform = waveform[:, :, : num_frames * model.total_stride]

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/sanity_{args.source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), waveform[0].cpu(), SAMPLE_RATE)

    with torch.no_grad():
        initial_recon = model(waveform)
        initial_loss = criterion(waveform, initial_recon)["total"].item()
    torchaudio.save(str(out_dir / "recon_step0.wav"), initial_recon[0].detach().cpu(), SAMPLE_RATE)

    losses = []
    step_times = []
    grad_norms = []
    best_loss = float("inf")
    best_step = -1
    for step in range(args.steps):
        t0 = time.perf_counter()

        recon = model(waveform)
        loss_dict = criterion(waveform, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step
            torchaudio.save(str(out_dir / "recon_best.wav"), recon[0].detach().cpu(), SAMPLE_RATE)

        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"step {step:>4d}  total {loss.item():.4f}  "
                f"wav {loss_dict['wav'].item():.4f}  mel {loss_dict['mel'].item():.4f}  "
                f"stft {loss_dict['stft'].item():.4f}  grad_norm {grad_norm.item():.3f}"
            )

    with torch.no_grad():
        final_recon = model(waveform)
    torchaudio.save(str(out_dir / f"recon_step{args.steps}.wav"), final_recon[0].detach().cpu(), SAMPLE_RATE)

    warmup = min(10, len(step_times) // 2)
    steady_step_times = step_times[warmup:]
    avg_step_time = sum(steady_step_times) / len(steady_step_times)

    print("\n--- summary ---")
    print(f"initial loss: {initial_loss:.4f}")
    print(f"final loss:   {losses[-1]:.4f}")
    print(f"best loss:    {best_loss:.4f}  (step {best_step})")
    print(f"reduction (final): {100 * (1 - losses[-1] / initial_loss):.1f}%")
    print(f"reduction (best):  {100 * (1 - best_loss / initial_loss):.1f}%")
    print(f"max grad norm (pre-clip): {max(grad_norms):.3f}")
    print(f"avg step time (steady-state): {avg_step_time * 1000:.1f} ms  ({1 / avg_step_time:.2f} steps/sec)")
    print(f"source={args.source} batch_size={args.batch_size} seconds={args.seconds} frames={num_frames}")
    print(f"samples written to: {out_dir}")


if __name__ == "__main__":
    main()
