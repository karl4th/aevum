#!/usr/bin/env python3
"""Benchmark training step time across batch sizes.

The Stage 1 encoder/decoder are sequential Python loops over ~100
recurrent steps/sec of audio. If wall-clock step time barely grows with
batch size, the bottleneck is kernel-launch/Python overhead rather than
compute, and batch size can be increased almost for free to raise dataset
throughput (audio-seconds processed per wall-clock second).

Usage:
    uv run scripts/benchmark_throughput.py
"""

import argparse
import time

import torch

from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss


def benchmark(model, criterion, optimizer, batch_size: int, seconds: float, device: torch.device, warmup: int, steps: int) -> float:
    num_frames = int(seconds * 100)
    waveform = torch.randn(batch_size, 1, model.total_stride * num_frames, device=device)

    for _ in range(warmup):
        recon = model(waveform)
        loss = criterion(waveform, recon)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        recon = model(waveform)
        loss = criterion(waveform, recon)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return elapsed / steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-size throughput sweep for Stage 1")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 16, 32, 64])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    print(f"{'batch_size':>10}  {'step_time_ms':>13}  {'audio_s/wall_s':>15}  {'peak_vram_mb':>12}")
    for batch_size in args.batch_sizes:
        model = DenseContinuousAutoencoder().to(device)
        criterion = ReconstructionLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        try:
            step_time = benchmark(model, criterion, optimizer, batch_size, args.seconds, device, args.warmup, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{batch_size:>10}  OOM")
            continue

        audio_per_wall_second = (batch_size * args.seconds) / step_time
        peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
        print(f"{batch_size:>10}  {step_time * 1000:>13.1f}  {audio_per_wall_second:>15.2f}  {peak_vram:>12.1f}")

        del model, criterion, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
