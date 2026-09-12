#!/usr/bin/env python3
"""Encoder-decoder interface diagnostic (Run 25 plan).

Test 2 (overfit_frozen_encoder_decoder.py) showed the frozen encoder's real
z_t still produces intermittent gradient explosions (grad_norm up to 3453)
feeding into decoder-v3, even though z itself never changes -- ruling out
"jointly-shifting distribution" as the sole explanation. This script asks a
sharper question: what specifically is different about the frozen encoder's
z compared to a latent decoder is already known to handle (the free-latent
Z from overfit_decoder_v3_free_latent.py), and does simply normalizing z
before the decoder fix the instability?

Step A: print latent_stats() for the frozen encoder's z once, up front
(compare by eye against the free-latent Z stats printed by
overfit_decoder_v3_free_latent.py at steps 0/100/500/1000/2000).

Steps D/E: apply one of three cheap, decoder-and-encoder-untouched
interventions to z before feeding it to decoder, controlled by --normalize:

  none        z' = z                                  (baseline, reproduces
                                                         overfit_frozen_encoder_decoder.py)
  rmsnorm     z' = RMSNorm(z)                          (learnable scale;
                                                         tests scale/conditioning)
  channelnorm z' = (z - mu_c) / (sigma_c + eps)         (fixed per-channel
                                                         whitening computed
                                                         once from the frozen
                                                         sequence itself --
                                                         diagnostic only, not
                                                         a production design;
                                                         tests anisotropy /
                                                         dead-or-hot channels)
  rmsnorm_linear  z' = W(RMSNorm(z))                    (learnable adapter --
                                                         decoder gets its own
                                                         latent space instead
                                                         of the encoder's
                                                         coordinates directly)

Step F: every --stats-every steps, log decoder_input_rms, per-branch decoder
hidden state RMS/abs-max (fast/mid/slow), and generator_input_rms -- to
catch which tensor moves first when a step is about to explode, instead of
only seeing the aggregate grad_norm/loss after the fact.

See docs/reports/stage1_v0.md, Run 25.

Usage:
    uv run scripts/diagnose_decoder_interface.py --normalize none --steps 2000
    uv run scripts/diagnose_decoder_interface.py --normalize rmsnorm --steps 2000
    uv run scripts/diagnose_decoder_interface.py --normalize channelnorm --steps 2000
    uv run scripts/diagnose_decoder_interface.py --normalize rmsnorm_linear --steps 2000
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
from aevum.utils.latent_stats import latent_stats

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
    """Fixed, non-trained encoder forward pass (identical construction to
    overfit_multiscale_encoder.py / overfit_frozen_encoder_decoder.py)."""
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


def decoder_forward_with_stats(
    decoder: ContinuousDecoder, u: torch.Tensor, dim: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Manually replicates ContinuousDecoder.forward, additionally returning
    stacked per-branch hidden states so their RMS/abs-max can be inspected."""
    batch_size = u.shape[0]
    state = decoder.initial_state(batch_size, device)
    y_steps, fast_steps, mid_steps, slow_steps = [], [], [], []
    for t in range(u.shape[1]):
        y_t, state = decoder.step(u[:, t, :], state)
        y_steps.append(y_t)
        fast_steps.append(state.fast)
        mid_steps.append(state.mid)
        slow_steps.append(state.slow)
    y = torch.stack(y_steps, dim=1)  # [B, T, dim]
    branch_states = {
        "fast": torch.stack(fast_steps, dim=1),
        "mid": torch.stack(mid_steps, dim=1),
        "slow": torch.stack(slow_steps, dim=1),
    }
    return y, branch_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the frozen-encoder-z -> decoder-v3 interface")
    parser.add_argument("--normalize", choices=["none", "rmsnorm", "channelnorm", "rmsnorm_linear"], default="none")
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
    parser.add_argument("--stats-every", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default=None)
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
    gates = {name: torch.tensor(args.gate_init, device=device) for name in BRANCH_TAU_RANGES}
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

    z_raw = compute_frozen_z(frontend, cells, projections, gates, w_x, target).transpose(1, 2)  # [B, T, dim]
    latent_stats("frozen_encoder_z (raw)", z_raw)

    normalizer: nn.Module | None = None
    channel_mean = channel_std = None
    if args.normalize == "rmsnorm":
        normalizer = nn.RMSNorm(dim).to(device)
    elif args.normalize == "channelnorm":
        channel_mean = z_raw.mean(dim=1, keepdim=True)
        channel_std = z_raw.std(dim=1, keepdim=True)
    elif args.normalize == "rmsnorm_linear":
        normalizer = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, dim)).to(device)

    def apply_normalize(z: torch.Tensor) -> torch.Tensor:
        if args.normalize == "none":
            return z
        if args.normalize == "channelnorm":
            return (z - channel_mean) / (channel_std + 1e-5)
        return normalizer(z)

    z_input = apply_normalize(z_raw)
    if args.normalize != "none":
        latent_stats(f"frozen_encoder_z ({args.normalize})", z_input.detach())

    trainable_modules = [decoder, w_skip_decoder, generator]
    if normalizer is not None:
        trainable_modules.append(normalizer)
    params = [p for m in trainable_modules for p in m.parameters()] + [gate_decoder]
    print(f"\nnormalize={args.normalize}  trainable params: {sum(p.numel() for p in params):,}")

    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/decoder_interface_{args.normalize}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    best_loss = float("inf")
    recon = None
    for step in range(args.steps):
        z_input = apply_normalize(z_raw)  # recomputed each step only to track a learnable normalizer's gradient
        y, branch_states = decoder_forward_with_stats(decoder, z_input, dim, device)
        y_for_generator = y.transpose(1, 2) + gate_decoder * w_skip_decoder(z_input.transpose(1, 2))
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

        if step % args.stats_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                decoder_input_rms = z_input.pow(2).mean().sqrt().item()
                generator_input_rms = y_for_generator.pow(2).mean().sqrt().item()
                branch_rms = {b: s.pow(2).mean().sqrt().item() for b, s in branch_states.items()}
                branch_absmax = {b: s.abs().max().item() for b, s in branch_states.items()}
            print(
                f"  [stats] step {step:>5d}  decoder_in_rms {decoder_input_rms:.4f}  "
                f"fast_h_rms {branch_rms['fast']:.4f}  mid_h_rms {branch_rms['mid']:.4f}  slow_h_rms {branch_rms['slow']:.4f}  "
                f"fast_h_absmax {branch_absmax['fast']:.4f}  mid_h_absmax {branch_absmax['mid']:.4f}  slow_h_absmax {branch_absmax['slow']:.4f}  "
                f"generator_in_rms {generator_input_rms:.4f}"
            )

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"\nsamples written to: {out_dir}")


if __name__ == "__main__":
    main()
