#!/usr/bin/env python3
"""Second isolation step: free learnable latent -> continuous decoder -> generator.

Run 1 (overfit_generator_only.py) proved `generator` alone can synthesize
this real clip perfectly from a free 100 Hz latent (user: "один в один").
This step adds `ContinuousDecoder` back in (its own norm+projection head is
already part of its forward pass) and removes only frontend/encoder: a free
per-frame latent Z_t in R^384 drives the decoder's continuous-time dynamics
directly, instead of a real encoder's z_t.

    learnable Z [1, T, 384] -> ContinuousDecoder -> y -> generator -> wav

  Result A (clean speech again): decoder + generator both exonerated.
  Bottleneck is narrowed to frontend -> encoder dynamics -> encoder head.

  Result B (noise again): generator already proved capable in isolation, so
  this points squarely at `ContinuousDecoder` -- specifically whether its
  leaky-integrator update `d_t = alpha*d_{t-1} + (1-alpha)*u_t` is
  over-smoothing information between steps.

See docs/reports/stage1_v0.md.

Usage:
    uv run scripts/overfit_decoder_generator.py --steps 4000
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.decoder import ContinuousDecoder
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit decoder + generator + free latent Z on one real clip")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="outputs/decoder_generator_only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    decoder = ContinuousDecoder(event_dim=args.latent_dim, output_dim=args.latent_dim).to(device)
    generator = CausalWaveformGenerator(input_dim=args.latent_dim).to(device)
    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    z = torch.nn.Parameter(0.01 * torch.randn(1, num_frames, args.latent_dim, device=device))
    param_count = sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in generator.parameters())
    print(f"free latent Z: {tuple(z.shape)} = {z.numel():,} parameters, plus decoder+generator's own {param_count:,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([*decoder.parameters(), *generator.parameters(), z], lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        y, _ = decoder(z)
        recon = generator(y.transpose(1, 2))
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*decoder.parameters(), *generator.parameters(), z], args.grad_clip_norm)
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
