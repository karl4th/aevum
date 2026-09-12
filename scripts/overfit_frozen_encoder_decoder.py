#!/usr/bin/env python3
"""Isolation test 2 (Run 22 plan): decoder capacity vs. encoder-decoder gradient coupling.

Test 1 (scripts/overfit_decoder_v3_free_latent.py) asks whether decoder v3
explodes given a *freely optimizable* input. This script asks a different
question: does decoder v3 explode given a *fixed* (not jointly trained)
real-audio-derived representation from the same fast+mid+slow+direct-
residual encoder architecture Run 19/22 already cleared?

The encoder (frontend + 3 ContinuousTimeCells, candidate_uses_hidden=False,
full adaptive tau, direct residual, gated fusion -- identical construction
to overfit_multiscale_encoder.py) is built at random initialization and
**never trained**: its forward pass runs under `torch.no_grad()`, so `z_t`
is a fixed function of the real audio clip, structurally similar to what a
real encoder produces but not adversarially co-adapted to the decoder
through any shared gradient. Only `ContinuousDecoder` + skip + `generator`
receive gradients from the reconstruction loss.

    real audio -> frontend -> {fast,mid,slow}+direct residual (FROZEN, no_grad)
               -> z_t -> ContinuousDecoder + skip -> generator -> wav

  Result A (stable): decoder v3 can learn to reconstruct from a fixed
  real-audio-shaped representation. Combined with test 1, this would mean
  the Run 20/21 explosions come specifically from the *joint* training
  dynamics -- encoder and decoder co-adapting/its z_t distribution shifting
  under shared backprop -- not decoder's forward capacity on its own.

  Result B (explodes): decoder can't handle even a fixed, non-shifting
  version of this kind of representation -- forward capacity is the
  problem regardless of coupling.

See docs/reports/stage1_v0.md, Run 22 (plan) / Run 24 (this test).

Usage:
    uv run scripts/overfit_frozen_encoder_decoder.py --steps 2000
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from torch import nn

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.decoder import ContinuousDecoder
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
    conv = nn.Conv1d(channels, channels, kernel_size=1).to(device)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(channels, device=device).unsqueeze(-1))
        conv.bias.zero_()
    return conv


def compute_frozen_z(
    frontend: nn.Module,
    cells: dict[str, ContinuousTimeCell],
    projections: dict[str, nn.Module],
    gates: dict[str, torch.Tensor],
    w_x: nn.Module,
    target: torch.Tensor,
) -> torch.Tensor:
    """Fixed, non-trained encoder forward pass -- identical construction to
    overfit_multiscale_encoder.py, but wrapped in no_grad since none of these
    modules are ever optimized in this script."""
    with torch.no_grad():
        features = frontend(target)  # [B, dim, T]
        states = {name: torch.zeros(1, BRANCH_HIDDEN_DIM, device=target.device) for name in cells}
        hidden_steps = {name: [] for name in cells}
        for t in range(features.shape[-1]):
            x_t = features[:, :, t]
            for name, cell in cells.items():
                states[name] = cell(x_t, states[name], dt=0.01)
                hidden_steps[name].append(states[name])
        hidden = {name: torch.stack(steps, dim=-1) for name, steps in hidden_steps.items()}

        z = w_x(features)
        for name in cells:
            z = z + gates[name] * projections[name](hidden[name])
    return z  # [B, dim, T], no grad history


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit decoder-v3 + generator on a FROZEN (untrained) encoder's real-audio z_t")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--gate-init", type=float, default=0.1)
    parser.add_argument("--decoder-gate-init", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="outputs/frozen_encoder_decoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    target = load_real_clip(args.data_root, args.librispeech_url, args.index, args.seconds, device)

    frontend = CausalAcousticFrontend().to(device).eval()
    dim = frontend.output_dim
    cells = {
        name: ContinuousTimeCell(dim, BRANCH_HIDDEN_DIM, *tau_range, candidate_uses_hidden=False).to(device).eval()
        for name, tau_range in BRANCH_TAU_RANGES.items()
    }
    projections = {name: nn.Conv1d(BRANCH_HIDDEN_DIM, dim, kernel_size=1).to(device).eval() for name in BRANCH_TAU_RANGES}
    gates = {name: torch.tensor(args.gate_init, device=device) for name in BRANCH_TAU_RANGES}  # fixed, not nn.Parameter
    w_x = identity_conv1d(dim, device).eval()
    for module in [frontend, w_x, *cells.values(), *projections.values()]:
        for p in module.parameters():
            p.requires_grad_(False)

    decoder = ContinuousDecoder(event_dim=dim, output_dim=dim).to(device)
    w_skip_decoder = identity_conv1d(dim, device)
    gate_decoder = nn.Parameter(torch.tensor(args.decoder_gate_init, device=device))
    generator = CausalWaveformGenerator(input_dim=dim).to(device)

    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    trainable_modules = [decoder, w_skip_decoder, generator]
    params = [p for m in trainable_modules for p in m.parameters()] + [gate_decoder]
    print(f"frozen encoder params: {sum(p.numel() for module in [frontend, w_x, *cells.values(), *projections.values()] for p in module.parameters()):,} (no grad)")
    print(f"trainable decoder-v3+generator params: {sum(p.numel() for p in params):,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    # z_t is a fixed function of the (frozen, untrained) encoder and the target audio --
    # compute it once, outside the training loop, since it never changes.
    z = compute_frozen_z(frontend, cells, projections, gates, w_x, target)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        y, _ = decoder(z.transpose(1, 2))  # [B, T, dim]
        y_for_generator = y.transpose(1, 2) + gate_decoder * w_skip_decoder(z)
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

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"\nsamples written to: {out_dir}")


if __name__ == "__main__":
    main()
