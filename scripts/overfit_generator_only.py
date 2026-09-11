#!/usr/bin/env python3
"""Cleanest possible test of generator capacity alone.

Throws out frontend, encoder, continuous dynamics, and decoder entirely.
Replaces them with a fully free, directly-learnable latent tensor
Z in R^{1 x latent_dim x T} (200 frames for a 2 s clip at 100 Hz -- one
independent 384-dim vector per frame, no bottleneck, no recurrence, no
encoder to blame). Trains Z jointly with `generator`'s own parameters to
memorize one real speech clip under the exact same reconstruction loss used
everywhere else in Stage 1.

    learnable Z [1, 384, 200] -> generator -> wav -> ReconstructionLoss

Binary question this answers: can `generator`, given the *easiest possible*
100 Hz input (no information bottleneck of any kind), synthesize this real
speech clip at all under our reconstruction loss?

  Result A: still noise after a few thousand steps -> the encoder /
  continuous dynamics / decoder stack is fully exonerated. The problem is
  either the generator's architecture or the reconstruction loss itself --
  see scripts/check_phase_problem.py to tell those two apart (does
  Griffin-Lim on the *target*'s true magnitude alone also sound bad?).

  Result B: clean/near-original speech -> generator is fine. The
  bottleneck is upstream, in what the encoder/continuous-dynamics/decoder
  chain does to the latent before it reaches the generator.

See docs/reports/stage1_v0.md.

Usage:
    uv run scripts/overfit_generator_only.py --steps 3000
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit generator + free latent Z on one real clip")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="outputs/generator_only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    generator = CausalWaveformGenerator(input_dim=args.latent_dim).to(device)
    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    z = torch.nn.Parameter(0.01 * torch.randn(1, args.latent_dim, num_frames, device=device))
    print(f"free latent Z: {tuple(z.shape)} = {z.numel():,} parameters, plus generator's own {sum(p.numel() for p in generator.parameters()):,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([*generator.parameters(), z], lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        recon = generator(z)
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*generator.parameters(), z], args.grad_clip_norm)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torchaudio.save(str(out_dir / "recon_best.wav"), recon[0].detach().cpu(), SAMPLE_RATE)

        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"step {step:>5d}  total {loss.item():.4f}  wav {loss_dict['wav'].item():.4f}  "
                f"mel {loss_dict['mel'].item():.4f}  stft {loss_dict['stft'].item():.4f}  "
                f"grad_norm {grad_norm.item():.3f}  best {best_loss:.4f}"
            )

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"\nsamples written to: {out_dir}")


if __name__ == "__main__":
    main()
