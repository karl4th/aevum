#!/usr/bin/env python3
"""Surgical isolation of *which* timescale branch destroys the signal.

Run 11 implicated the encoder's fast/mid/slow continuous-time recurrence as
a whole (it diverged on real audio; Run 12's frontend+linear-projection
alternative was clean). This step tests one branch alone, no cross-timescale
connections at all:

    real audio -> frontend -> ONE ContinuousTimeCell (fast|mid|slow) -> per-frame projection -> generator -> wav

Hypothesis (user, grounded in the tau/alpha values actually measured in Run
6/7): alpha_fast~0.80 (20% new information per 10ms step) should still
permit reasonable reconstruction; alpha_mid~0.964 (3.6%/step) should be
noticeably worse; alpha_slow~0.996 (0.4%/step) should turn speech to mush --
because a state built to hold slowly-changing context is being forced to
also be the *only* transport channel for fast-changing acoustic detail, and
those are different jobs.

Usage:
    uv run scripts/overfit_single_branch.py --branch fast --steps 1000
    uv run scripts/overfit_single_branch.py --branch mid  --steps 1000
    uv run scripts/overfit_single_branch.py --branch slow --steps 1000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from torch import nn

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.dynamics.cell import ContinuousTimeCell
from aevum.models.frontend import CausalAcousticFrontend
from aevum.models.generator import CausalWaveformGenerator
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000

# Same tau ranges as aevum.models.dynamics.multiscale.MultiTimescaleDynamics's default.
BRANCH_TAU_RANGES = {
    "fast": (0.010, 0.080),
    "mid": (0.050, 0.500),
    "slow": (0.300, 5.000),
}
BRANCH_HIDDEN_DIM = 192


def load_real_clip(data_root: str, url: str, index: int, seconds: float, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]  # [1, samples] -- same fixed clip used everywhere else in this report
    return clip.unsqueeze(0).to(device)  # [1, 1, samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a single continuous-time branch + generator on one real clip")
    parser.add_argument("--branch", choices=["fast", "mid", "slow"], required=True)
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    frontend = CausalAcousticFrontend().to(device)
    tau_min, tau_max = BRANCH_TAU_RANGES[args.branch]
    cell = ContinuousTimeCell(frontend.output_dim, BRANCH_HIDDEN_DIM, tau_min, tau_max).to(device)
    projection = nn.Conv1d(BRANCH_HIDDEN_DIM, frontend.output_dim, kernel_size=1).to(device)
    generator = CausalWaveformGenerator(input_dim=frontend.output_dim).to(device)

    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    modules = [frontend, cell, projection, generator]
    param_count = sum(p.numel() for m in modules for p in m.parameters())
    print(f"branch={args.branch} tau_range=({tau_min},{tau_max}) hidden_dim={BRANCH_HIDDEN_DIM} params={param_count:,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=args.lr)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/single_branch_{args.branch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        features = frontend(target)  # [B, 384, T]
        state = torch.zeros(1, BRANCH_HIDDEN_DIM, device=device)
        hidden_steps = []
        for t in range(features.shape[-1]):
            state = cell(features[:, :, t], state, dt=0.01)
            hidden_steps.append(state)
        hidden = torch.stack(hidden_steps, dim=-1)  # [B, hidden_dim, T]

        z = projection(hidden)
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
