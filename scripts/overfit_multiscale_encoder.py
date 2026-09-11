#!/usr/bin/env python3
"""Assemble the redesigned multi-timescale encoder on one real clip.

Three independent timescale branches (no cross-connections yet) plus a
direct instantaneous path, combined via a small-init gated fusion:

    f_t --W_x---------------------------------------------------┐
    f_t --fast ContinuousTimeCell--> P_F --* g_F-----------------┤
    f_t --mid  ContinuousTimeCell--> P_M --* g_M-----------------┼--> z_t --> generator
    f_t --slow ContinuousTimeCell--> P_S --* g_S-----------------┘

Every branch uses candidate_uses_hidden=False (Run 14: no evidence the
candidate's own W_hh h_{t-1} term is needed once memory already exists via
alpha_t*h_{t-1}) and full adaptive tau (Run 17/18: safe once a branch isn't
also forced to be the sole transport channel). No cross-timescale
connections yet (tech_spec.md section 9) -- this proves three independent
branches plus the direct path work together before adding communication
between them.

Gates g_F/g_M/g_S are learnable scalars initialized small (default 0.1),
and W_x starts at the identity, so training begins close to the
already-proven frontend -> generator baseline (Run 12) and the temporal
branches have to earn their contribution rather than being relied on from
step 0.

See docs/reports/stage1_v0.md, Run 19.

Usage:
    uv run scripts/overfit_multiscale_encoder.py --steps 2000
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


def identity_conv1d(channels: int, device: torch.device) -> nn.Conv1d:
    """A Conv1d(channels, channels, kernel_size=1) initialized to the identity map."""
    conv = nn.Conv1d(channels, channels, kernel_size=1).to(device)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(channels, device=device).unsqueeze(-1))
        conv.bias.zero_()
    return conv


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit the full fast+mid+slow encoder + gated fusion + generator")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--gate-init", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="outputs/multiscale_encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    frontend = CausalAcousticFrontend().to(device)
    dim = frontend.output_dim

    cells = {
        name: ContinuousTimeCell(dim, BRANCH_HIDDEN_DIM, *tau_range, candidate_uses_hidden=False).to(device)
        for name, tau_range in BRANCH_TAU_RANGES.items()
    }
    projections = {name: nn.Conv1d(BRANCH_HIDDEN_DIM, dim, kernel_size=1).to(device) for name in BRANCH_TAU_RANGES}
    gates = {
        name: nn.Parameter(torch.tensor(args.gate_init, device=device)) for name in BRANCH_TAU_RANGES
    }
    w_x = identity_conv1d(dim, device)
    generator = CausalWaveformGenerator(input_dim=dim).to(device)

    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    modules = [frontend, w_x, generator, *cells.values(), *projections.values()]
    params = [p for m in modules for p in m.parameters()] + list(gates.values())
    param_count = sum(p.numel() for p in params)
    print(f"gate_init={args.gate_init} params={param_count:,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        features = frontend(target)  # [B, dim, T]
        states = {name: torch.zeros(1, BRANCH_HIDDEN_DIM, device=device) for name in cells}
        hidden_steps = {name: [] for name in cells}
        for t in range(features.shape[-1]):
            x_t = features[:, :, t]
            for name, cell in cells.items():
                states[name] = cell(x_t, states[name], dt=0.01)
                hidden_steps[name].append(states[name])
        hidden = {name: torch.stack(steps, dim=-1) for name, steps in hidden_steps.items()}  # [B, hidden_dim, T]

        z = w_x(features)
        for name in cells:
            z = z + gates[name] * projections[name](hidden[name])

        recon = generator(z)
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
            gate_values = " ".join(f"g_{name}={gates[name].item():.4f}" for name in cells)
            print(
                f"step {step:>5d}  total {loss.item():.4f}  wav {loss_dict['wav'].item():.4f}  "
                f"mel {loss_dict['mel'].item():.4f}  stft {loss_dict['stft'].item():.4f}  "
                f"grad_norm {grad_norm.item():.3f}  best {best_loss:.4f}  {gate_values}"
            )

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"\nsamples written to: {out_dir}")


if __name__ == "__main__":
    main()
