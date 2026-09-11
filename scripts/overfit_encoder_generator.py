#!/usr/bin/env python3
"""Third isolation step: real audio -> frontend -> encoder -> generator, no decoder.

Runs 9 and 10 cleared `generator` and `ContinuousDecoder` in isolation, both
fed a free per-frame latent. This step removes the free-latent "cheat" and
puts `ContinuousEncoder` (frontend + fast/mid/slow dynamics + head) back to
work analyzing the real waveform directly, feeding `generator` with no
decoder in between at all:

    real audio -> frontend -> encoder dynamics -> encoder head -> generator -> wav

If this also reproduces the clip cleanly, the emerging picture is:

    free latent -> generator                 (Run 9)  clean
    free latent -> decoder -> generator       (Run 10) clean
    audio -> encoder -> generator             (this)   clean?
    audio -> encoder -> decoder -> generator  (full)   noise

...which would point specifically at the encoder-decoder *interface* (the
decoder's leaky-integrator dynamics smoothing/distorting what an otherwise
capable encoder hands it), not at any single block's raw capacity.

See docs/reports/stage1_v0.md.

Usage:
    uv run scripts/overfit_encoder_generator.py --steps 3000
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.encoder import ContinuousEncoder
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit encoder + generator (no decoder) on one real clip")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="outputs/encoder_generator_only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    encoder = ContinuousEncoder(latent_dim=args.latent_dim).to(device)
    generator = CausalWaveformGenerator(input_dim=args.latent_dim).to(device)
    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    param_count = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in generator.parameters())
    print(f"no free latent this time -- encoder analyzes real audio. encoder+generator params: {param_count:,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *generator.parameters()], lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        z, _ = encoder(target)
        recon = generator(z.transpose(1, 2))
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *generator.parameters()], args.grad_clip_norm)
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
