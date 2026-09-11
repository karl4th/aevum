#!/usr/bin/env python3
"""Fourth isolation step: frontend -> per-frame linear projection -> generator.

Run 11 (encoder + generator, real audio, no decoder) diverged: grad_norm
spiked 378 -> 1650 -> 642 -> 982 across steps 150-225 and loss got *worse*
than at step 0. This is new -- Runs 9/10 (free latent, no real encoder
recurrence involved) never showed this. This step removes only the
fast/mid/slow continuous-time recurrence, replacing it with a trivial
per-frame linear projection (no cross-timestep dependency at all beyond
what the causal frontend already introduces):

    real audio -> frontend -> Linear(384, 384) per frame -> generator -> wav

  Result A (fast, clean, no explosion): frontend and generator are both
  fine even on real audio; the problem is specifically the encoder's
  continuous-time recurrent dynamics (the fast/mid/slow cells) -- time to
  take ContinuousTimeCell apart directly.

  Result B (explodes/noisy again): the problem is upstream of any
  recurrence at all -- the frontend itself, its normalization, stride/
  alignment, or the frontend->generator interface.

See docs/reports/stage1_v0.md.

Usage:
    uv run scripts/overfit_frontend_generator.py --steps 2000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from torch import nn

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.frontend import CausalAcousticFrontend
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit frontend + linear projection + generator (no encoder recurrence)")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="outputs/frontend_generator_only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    frontend = CausalAcousticFrontend().to(device)
    projection = nn.Conv1d(frontend.output_dim, frontend.output_dim, kernel_size=1).to(device)  # per-frame Linear
    generator = CausalWaveformGenerator(input_dim=frontend.output_dim).to(device)
    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    modules = [frontend, projection, generator]
    param_count = sum(p.numel() for m in modules for p in m.parameters())
    print(f"no recurrence, no free latent -- frontend+projection+generator params: {param_count:,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        z = projection(frontend(target))
        recon = generator(z)
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for m in modules for p in m.parameters()], args.grad_clip_norm)
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
