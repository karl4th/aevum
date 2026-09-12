#!/usr/bin/env python3
"""Isolation test 1 (Run 22 plan): does decoder v3 (with skip) explode on its own?

Run 10 showed a *plain* `ContinuousDecoder` (no skip) trains cleanly from a
free latent, but only up to ~400 steps. Runs 20/21 (encoder+decoder v3, full
pipeline) both explode well before 2000 steps, well after looking clean
initially -- so "looked stable early" is no longer sufficient evidence.

This script re-tests the *v3* decoder (`ContinuousDecoder` + the direct
skip added in Run 21: `y_for_generator = decoder(z) + gate*W_skip(z)`) fed a
fully free, directly-learnable per-frame latent, run to the full 2000
steps:

    learnable Z [1, T, 384] -> ContinuousDecoder + skip -> generator -> wav

  Result A (stable for the full 2000 steps): ContinuousDecoder (+skip) has
  no inherent forward-dynamics problem -- the Run 20/21 explosions must come
  from the *interaction* with the encoder's evolving z_t distribution
  (gradient coupling), not decoder capacity alone. Next: test 2
  (frozen encoder -> decoder, scripts/overfit_frozen_encoder_decoder.py).

  Result B (explodes): decoder itself cannot sustain training even with an
  unconstrained, freely-optimizable input -- the problem is in
  ContinuousDecoder's own dynamics, independent of what feeds it.

See docs/reports/stage1_v0.md, Run 22 (plan) / Run 23 (this test).

Usage:
    uv run scripts/overfit_decoder_v3_free_latent.py --steps 2000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from torch import nn

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.decoder import ContinuousDecoder
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss
from aevum.utils.latent_stats import latent_stats

SAMPLE_RATE = 24_000


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def identity_conv1d(channels: int, device: torch.device) -> nn.Conv1d:
    conv = nn.Conv1d(channels, channels, kernel_size=1).to(device)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(channels, device=device).unsqueeze(-1))
        conv.bias.zero_()
    return conv


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit decoder-v3 (with skip) + generator + free latent Z on one real clip")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--decoder-gate-init", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--stats-steps",
        type=int,
        nargs="+",
        default=[0, 100, 500, 1000, 2000],
        help="training steps at which to print latent_stats(Z) -- used to compare against the frozen encoder's z (Run 25)",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/decoder_v3_free_latent")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    decoder = ContinuousDecoder(event_dim=args.latent_dim, output_dim=args.latent_dim).to(device)
    w_skip_decoder = identity_conv1d(args.latent_dim, device)
    gate_decoder = nn.Parameter(torch.tensor(args.decoder_gate_init, device=device))
    generator = CausalWaveformGenerator(input_dim=args.latent_dim).to(device)

    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    z = torch.nn.Parameter(0.01 * torch.randn(1, num_frames, args.latent_dim, device=device))
    modules = [decoder, w_skip_decoder, generator]
    params = [p for m in modules for p in m.parameters()] + [z, gate_decoder]
    print(f"free latent Z: {tuple(z.shape)} = {z.numel():,} parameters, plus decoder-v3+generator's own {sum(p.numel() for m in modules for p in m.parameters()):,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    stats_steps = set(args.stats_steps)
    for step in range(args.steps):
        if step in stats_steps:
            latent_stats(f"free_Z_step{step}", z)

        y, _ = decoder(z)  # [B, T, latent_dim]
        y_for_generator = y.transpose(1, 2) + gate_decoder * w_skip_decoder(z.transpose(1, 2))
        recon = generator(y_for_generator)
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torchaudio.save(str(out_dir / "recon_best.wav"), recon[0].detach().cpu(), SAMPLE_RATE)

        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"step {step:>5d}  total {loss.item():.4f}  wav {loss_dict['wav'].item():.4f}  "
                f"mel {loss_dict['mel'].item():.4f}  stft {loss_dict['stft'].item():.4f}  "
                f"grad_norm {grad_norm.item():.3f}  best {best_loss:.4f}  g_decoder={gate_decoder.item():.4f}"
            )

    if args.steps in stats_steps:
        latent_stats(f"free_Z_step{args.steps}", z)

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"\nsamples written to: {out_dir}")


if __name__ == "__main__":
    main()
