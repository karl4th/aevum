#!/usr/bin/env python3
"""Diagnose where the huge pre-clip gradient norms come from.

Context (docs/reports/stage1_v0.md): a single-clip overfit test on a real
LibriSpeech utterance reached a lower loss than any previous run (79%
reduction) but produced noisy, unintelligible audio, and its max pre-clip
grad norm (885) was ~8.5x higher than the same test on a smooth synthetic
tone (104). The suspected mechanism (tech_spec.md section 7): the adaptive
time constant's sensitivity `d(alpha)/d(tau) = exp(-dt/tau) * dt/tau^2` blows
up as tau approaches tau_min, and analytically this is ~340x larger for the
fast branch (tau_min=0.010s) than the slow branch (tau_min=0.300s) --- real
speech's sharp transients (plosives, onsets) are exactly the kind of signal
that would push the fast branch's tau toward that boundary, unlike the
smooth synthetic tone.

This script logs, every few steps: (a) gradient norm per architectural
block/branch -- in particular splitting each cell's `time_constant` (tau)
parameters from its `candidate` parameters, and (b) the tau/alpha value
distribution per branch, to confirm or refute that hypothesis directly
rather than guessing from the aggregate grad norm alone.

Usage:
    uv run scripts/diagnose_gradients.py --source real --steps 200
"""

import argparse

import torch

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss

SAMPLE_RATE = 24_000


def make_synthetic_clip(seconds: float, batch_size: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(int(seconds * SAMPLE_RATE), device=device) / SAMPLE_RATE
    f0 = 140.0 + 40.0 * torch.sin(2 * torch.pi * 0.7 * t)
    envelope = 0.5 * (1 + torch.sin(2 * torch.pi * 1.3 * t))
    phase = 2 * torch.pi * torch.cumsum(f0, dim=0) / SAMPLE_RATE
    signal = envelope * (torch.sin(phase) + 0.3 * torch.sin(3 * phase) + 0.15 * torch.sin(5 * phase))
    signal = signal / signal.abs().max()
    return signal.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)


def load_real_clip(data_root: str, url: str, index: int, seconds: float, batch_size: int, device: torch.device) -> torch.Tensor:
    dataset = LibriSpeechSegments(root=data_root, url=url, segment_seconds=seconds, download=True)
    torch.manual_seed(0)
    clip = dataset[index]
    return clip.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


# (group_name, substring_predicate) -- checked in order, first match wins.
_GROUP_RULES = [
    ("encoder.frontend", lambda n: n.startswith("encoder.frontend")),
    ("encoder.fast.tau", lambda n: "encoder.dynamics.fast_cell.time_constant" in n),
    ("encoder.fast.candidate", lambda n: "encoder.dynamics.fast_cell.candidate" in n),
    ("encoder.mid.tau", lambda n: "encoder.dynamics.mid_cell.time_constant" in n),
    ("encoder.mid.candidate", lambda n: "encoder.dynamics.mid_cell.candidate" in n),
    ("encoder.slow.tau", lambda n: "encoder.dynamics.slow_cell.time_constant" in n),
    ("encoder.slow.candidate", lambda n: "encoder.dynamics.slow_cell.candidate" in n),
    ("encoder.cross", lambda n: n.startswith("encoder.dynamics.") and "cell" not in n),
    ("encoder.head", lambda n: n.startswith(("encoder.norm", "encoder.to_latent"))),
    ("decoder.fast.tau", lambda n: "decoder.dynamics.fast_cell.time_constant" in n),
    ("decoder.fast.candidate", lambda n: "decoder.dynamics.fast_cell.candidate" in n),
    ("decoder.mid.tau", lambda n: "decoder.dynamics.mid_cell.time_constant" in n),
    ("decoder.mid.candidate", lambda n: "decoder.dynamics.mid_cell.candidate" in n),
    ("decoder.slow.tau", lambda n: "decoder.dynamics.slow_cell.time_constant" in n),
    ("decoder.slow.candidate", lambda n: "decoder.dynamics.slow_cell.candidate" in n),
    ("decoder.cross", lambda n: n.startswith("decoder.dynamics.") and "cell" not in n),
    ("decoder.head", lambda n: n.startswith(("decoder.norm", "decoder.to_output"))),
    ("generator", lambda n: n.startswith("generator")),
]


def grad_norm_groups(model: torch.nn.Module) -> dict[str, float]:
    sums = {name: 0.0 for name, _ in _GROUP_RULES}
    sums["other"] = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = next((g for g, predicate in _GROUP_RULES if predicate(name)), "other")
        sums[group] += param.grad.detach().pow(2).sum().item()
    return {group: total**0.5 for group, total in sums.items() if total > 0}


def tau_alpha_stats(cell: torch.nn.Module, name: str) -> str:
    stats = cell.stop_recording()
    tau, alpha = stats["tau"], stats["alpha"]
    if tau.numel() == 0:
        return f"{name:>18}  (no data)"
    q = torch.tensor([0.0, 0.5, 0.95, 1.0], device=tau.device)
    tau_q = torch.quantile(tau, q)
    alpha_q = torch.quantile(alpha, q)
    return (
        f"{name:>18}  tau[min={tau_q[0]:.4f} med={tau_q[1]:.4f} p95={tau_q[2]:.4f} max={tau_q[3]:.4f}]"
        f"  alpha[min={alpha_q[0]:.4f} med={alpha_q[1]:.4f} p95={alpha_q[2]:.4f} max={alpha_q[3]:.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-block gradient norm and tau/alpha diagnostics")
    parser.add_argument("--source", choices=["synthetic", "real"], default="real")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--real-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    model = DenseContinuousAutoencoder().to(device)
    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cells = {
        "encoder.fast": model.encoder.dynamics.fast_cell,
        "encoder.mid": model.encoder.dynamics.mid_cell,
        "encoder.slow": model.encoder.dynamics.slow_cell,
        "decoder.fast": model.decoder.dynamics.fast_cell,
        "decoder.mid": model.decoder.dynamics.mid_cell,
        "decoder.slow": model.decoder.dynamics.slow_cell,
    }

    if args.source == "real":
        waveform = load_real_clip(args.data_root, args.librispeech_url, args.real_index, args.seconds, args.batch_size, device)
    else:
        waveform = make_synthetic_clip(args.seconds, args.batch_size, device)
    num_frames = waveform.shape[-1] // model.total_stride
    waveform = waveform[:, :, : num_frames * model.total_stride]

    for step in range(args.steps):
        do_log = step % args.log_every == 0 or step == args.steps - 1
        if do_log:
            for cell in cells.values():
                cell.start_recording()

        recon = model(waveform)
        loss_dict = criterion(waveform, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()

        if do_log:
            print(f"\n=== step {step} === total {loss.item():.4f}")
            group_norms = grad_norm_groups(model)
            for group, norm in sorted(group_norms.items(), key=lambda kv: -kv[1]):
                print(f"  grad_norm[{group:>22}] = {norm:9.3f}")
            for name, cell in cells.items():
                print("  " + tau_alpha_stats(cell, name))

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()


if __name__ == "__main__":
    main()
