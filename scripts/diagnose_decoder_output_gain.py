#!/usr/bin/env python3
"""Decompose generator_in and test three ablations of the decoder's output path.

Run 25 (diagnose_decoder_interface.py, --normalize rmsnorm) showed that even
with the frozen encoder's z rescaled to rms~1.0, decoder-v3 + generator still
explodes intermittently. Critically: `fast_h`/`mid_h` saturate near the
tanh bound (|h| <= 1, which is provably bounded since
`h_t = alpha*h_{t-1} + (1-alpha)*tanh(...)` is a convex combination of two
things in [-1,1]) almost immediately and *stay* saturated -- ruling out
unbounded hidden-state growth as the exploding mechanism. Meanwhile
`generator_in_rms` keeps growing unboundedly (0.6 -> 2.2+) in lockstep with
the large loss/grad_norm spikes.

This means the runaway amplitude must come from *after* the bounded hidden
state: `ContinuousDecoder.to_output` (an unconstrained Linear layer) and/or
the learnable `gate_decoder` scaling the skip-connection design from Run 21.
Bounded `h` can still produce arbitrarily large `to_output(h)` if the
Linear layer's weights grow.

This script decomposes:

    generator_in = skip(z) + gate_decoder * temporal_raw(decoder(z))

logging, every --log-every steps, as one JSON record each: the usual loss/
grad_norm/best, `to_output`'s weight norm/abs-max, rms/abs-max of
`skip`/`temporal_raw`/`temporal_gated`/`generator_in`, per-branch hidden
state AND pre-tanh-candidate rms/abs-max (fast/mid/slow), the generator's
pre-final-tanh rms/abs-max, and the final waveform's rms and saturation
fraction (`(|wav| > 0.95).mean()`).

--ablation selects which of the user's three control experiments to run,
always with the same full logging:

  baseline     gate_decoder trainable (init 0.1), to_output unconstrained --
               reproduces the observed instability with full instrumentation.
  freeze_gate  gate_decoder fixed at 0.1 (excluded from the optimizer). If
               generator_in still grows -> the gate isn't the driver,
               to_output's own weight growth is (Variant A).
  fixed_gain   temporal_raw is rescaled (using a *detached* RMS statistic,
               not a learnable norm) to a fixed target RMS before being
               gated -- diagnostic only, not a production design. If this
               removes the spikes -> unconstrained downstream gain is the
               mechanism (matches the user's saturation -> gain-compensation
               -> generator-saturation hypothesis).
  no_temporal  gate_decoder fixed at 0.0 (excluded from the optimizer):
               only skip(z) reaches the generator; decoder's own dynamics
               still compute (and its parameters still exist) but never
               contribute to the output. If this is stable for the full run,
               the temporal/to_output branch is definitively the culprit.

See docs/reports/stage1_v0.md, Run 26.

Run 26 (--ablation all, or equivalently the four separate commands) found
that only `freeze_gate` (gate fixed at 0.1) sounds natural on listening;
`baseline`, `fixed_gain`, and `no_temporal` all sound robotic despite
`no_temporal` having the second-best loss -- suggesting the perceptual
optimum for the gate value may not coincide with the loss optimum. Run 27's
`--gate-values` sweep tests several fixed gate values in one command (same
mechanics as `freeze_gate`, parameterized) to map out `loss` vs `g` and let
the user listen to each.

Usage (each capped at 1000 steps per the user's instruction):
    uv run scripts/diagnose_decoder_output_gain.py --steps 1000
    uv run scripts/diagnose_decoder_output_gain.py --ablation freeze_gate --steps 1000
    uv run scripts/diagnose_decoder_output_gain.py --gate-values 0.05 0.10 0.15 0.20 --steps 1000
"""

import argparse
import json
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
    decoder: ContinuousDecoder, u: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Runs ``decoder.step_branches`` (the same fusion building blocks
    ``ContinuousDecoder.step`` uses in production) and returns the temporal
    branch, the skip branch, and stacked per-branch hidden states -- both raw,
    un-gated/un-fused, so this diagnostic can apply its own gate ablations
    without re-deriving the decoder's fusion formula."""
    batch_size = u.shape[0]
    state = decoder.initial_state(batch_size, device)
    temporal_steps, skip_steps, fast_steps, mid_steps, slow_steps = [], [], [], [], []
    for t in range(u.shape[1]):
        temporal_t, skip_t, state = decoder.step_branches(u[:, t, :], state)
        temporal_steps.append(temporal_t)
        skip_steps.append(skip_t)
        fast_steps.append(state.fast)
        mid_steps.append(state.mid)
        slow_steps.append(state.slow)
    temporal_raw = torch.stack(temporal_steps, dim=1)  # [B, T, dim]
    skip = torch.stack(skip_steps, dim=1)  # [B, T, dim]
    branch_states = {
        "fast": torch.stack(fast_steps, dim=1),
        "mid": torch.stack(mid_steps, dim=1),
        "slow": torch.stack(slow_steps, dim=1),
    }
    return temporal_raw, skip, branch_states


def generator_forward_with_pretanh(generator: CausalWaveformGenerator, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicates CausalWaveformGenerator.forward, also returning the pre-tanh logits."""
    x = generator.in_proj(y)
    for block in generator.blocks:
        x = block(x)
    pre_tanh = generator.to_wave(x)
    return torch.tanh(pre_tanh), pre_tanh


def rms(x: torch.Tensor) -> float:
    return x.detach().float().pow(2).mean().sqrt().item()


def absmax(x: torch.Tensor) -> float:
    return x.detach().float().abs().max().item()


ALL_ABLATIONS = ["baseline", "freeze_gate", "fixed_gain", "no_temporal"]


def run_ablation(args: argparse.Namespace, ablation: str, gate_override: float | None = None) -> dict:
    """Runs one ablation to completion and returns its final summary stats.

    ``gate_override`` (used by the ``--gate-values`` sweep) fixes the gate at
    this exact value, excluded from the optimizer -- the same mechanics as
    ``freeze_gate``, just parameterized, and labeled distinctly in output
    paths/logs so a sweep's runs don't collide with the four named ablations.
    """
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
    encoder_gates = {name: torch.tensor(args.gate_init, device=device) for name in BRANCH_TAU_RANGES}
    w_x = identity_conv1d(dim, device).eval()
    for module in [frontend, w_x, *cells.values(), *projections.values()]:
        for p in module.parameters():
            p.requires_grad_(False)

    decoder = ContinuousDecoder(
        event_dim=dim,
        output_dim=dim,
        candidate_uses_hidden=not args.no_decoder_self_recurrence,
        adaptive_tau=not args.decoder_fixed_tau,
        disable_cross_connections=args.disable_decoder_cross_connections,
    ).to(device)
    decoder_cells = {"fast": decoder.dynamics.fast_cell, "mid": decoder.dynamics.mid_cell, "slow": decoder.dynamics.slow_cell}
    generator = CausalWaveformGenerator(input_dim=dim).to(device)

    num_frames = target.shape[-1] // generator.total_stride
    target = target[:, :, : num_frames * generator.total_stride]

    z_raw = compute_frozen_z(frontend, cells, projections, encoder_gates, w_x, target)  # [B, dim, T]

    if gate_override is not None:
        gate_value, gate_trainable, label = gate_override, False, f"gate_{gate_override:.2f}"
    elif ablation == "no_temporal":
        gate_value, gate_trainable, label = 0.0, False, ablation
    elif ablation == "freeze_gate":
        gate_value, gate_trainable, label = args.decoder_gate_init, False, ablation
    else:  # baseline, fixed_gain
        gate_value, gate_trainable, label = args.decoder_gate_init, True, ablation

    if args.no_decoder_self_recurrence:
        label += "_no_dec_self_rec"
    if args.decoder_fixed_tau:
        label += "_dec_fixed_tau"
    if args.disable_decoder_cross_connections:
        label += "_no_dec_cross"
    if args.gate_branch == "skip":
        label += "_gate_on_skip"  # non-default: matches production's decoder.step fusion

    gate_decoder = nn.Parameter(torch.tensor(gate_value, device=device)) if gate_trainable else torch.tensor(gate_value, device=device)

    trainable_modules = [decoder, generator]
    params = [p for m in trainable_modules for p in m.parameters()]
    if gate_trainable:
        params.append(gate_decoder)
    print(f"\n=== {label} ===  trainable params: {sum(p.numel() for p in params):,}")

    alignment_delay = generator.total_stride if args.alignment_delay is None else args.alignment_delay
    criterion = ReconstructionLoss(delay=alignment_delay).to(device)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/decoder_output_gain_{label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "target.wav"), target[0].detach().cpu(), SAMPLE_RATE)

    log_path = out_dir / "log.json"
    log_records: list[dict] = []

    best_loss = float("inf")
    recon = None
    z_bt = z_raw.transpose(1, 2)  # [B, T, dim], fixed input to decoder
    for step in range(args.steps):
        do_log = step % args.log_every == 0 or step == args.steps - 1
        if do_log:
            for cell in decoder_cells.values():
                cell.start_recording()

        temporal_raw, skip, branch_states = decoder_forward_with_stats(decoder, z_bt, device)  # [B, T, dim] each

        if ablation == "fixed_gain":
            raw_rms = temporal_raw.detach().pow(2).mean().sqrt().clamp_min(1e-8)
            temporal_for_gate = temporal_raw * (args.fixed_gain_target / raw_rms)
        else:
            temporal_for_gate = temporal_raw

        # args.gate_branch picks WHICH branch gate_decoder scales -- ContinuousDecoder.fuse is
        # the same production formula either way (fuse(base, x, gate) = base + gate*x), called
        # with the two operands swapped depending on which one is meant to be gated. Production
        # (decoder.step) always gates skip; this script defaults to gating temporal (its
        # original, historical ablation design -- see module docstring's "no_temporal" gate=0.0
        # meaning "only skip(z) reaches the generator") -- the two are NOT equivalent at the
        # same gate value, so which one is active is always logged and labeled explicitly.
        if args.gate_branch == "skip":
            temporal_gated = temporal_for_gate  # logged/reported name kept for output compatibility
            generator_in = ContinuousDecoder.fuse(temporal_for_gate, skip, gate_decoder).transpose(1, 2)
        else:
            temporal_gated = gate_decoder * temporal_for_gate
            generator_in = ContinuousDecoder.fuse(skip, temporal_for_gate, gate_decoder).transpose(1, 2)

        recon, pre_tanh = generator_forward_with_pretanh(generator, generator_in)
        loss_dict = criterion(target, recon)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torchaudio.save(str(out_dir / "recon_best.wav"), recon[0].detach().cpu(), SAMPLE_RATE)

        if do_log:
            cell_stats = {name: cell.stop_recording() for name, cell in decoder_cells.items()}
            record = {
                "step": step,
                "total_loss": loss.item(),
                "wav_loss": loss_dict["wav"].item(),
                "mel_loss": loss_dict["mel"].item(),
                "stft_loss": loss_dict["stft"].item(),
                "grad_norm": grad_norm.item(),
                "best_loss": best_loss,
                "g_decoder": gate_decoder.item(),
                "decoder_in_rms": rms(z_bt),
                "fast_h_rms": rms(branch_states["fast"]),
                "mid_h_rms": rms(branch_states["mid"]),
                "slow_h_rms": rms(branch_states["slow"]),
                "fast_h_absmax": absmax(branch_states["fast"]),
                "mid_h_absmax": absmax(branch_states["mid"]),
                "slow_h_absmax": absmax(branch_states["slow"]),
                "fast_pretanh_rms": rms(cell_stats["fast"]["candidate_pretanh"]) if cell_stats["fast"]["candidate_pretanh"].numel() else None,
                "mid_pretanh_rms": rms(cell_stats["mid"]["candidate_pretanh"]) if cell_stats["mid"]["candidate_pretanh"].numel() else None,
                "slow_pretanh_rms": rms(cell_stats["slow"]["candidate_pretanh"]) if cell_stats["slow"]["candidate_pretanh"].numel() else None,
                "fast_pretanh_absmax": absmax(cell_stats["fast"]["candidate_pretanh"]) if cell_stats["fast"]["candidate_pretanh"].numel() else None,
                "mid_pretanh_absmax": absmax(cell_stats["mid"]["candidate_pretanh"]) if cell_stats["mid"]["candidate_pretanh"].numel() else None,
                "slow_pretanh_absmax": absmax(cell_stats["slow"]["candidate_pretanh"]) if cell_stats["slow"]["candidate_pretanh"].numel() else None,
                "skip_rms": rms(skip),
                "skip_absmax": absmax(skip),
                "temporal_raw_rms": rms(temporal_raw),
                "temporal_raw_absmax": absmax(temporal_raw),
                "temporal_gated_rms": rms(temporal_gated),
                "temporal_gated_absmax": absmax(temporal_gated),
                "to_output_weight_norm": decoder.to_output.weight.norm().item(),
                "to_output_weight_absmax": decoder.to_output.weight.abs().max().item(),
                "generator_in_rms": rms(generator_in),
                "generator_in_absmax": absmax(generator_in),
                "generator_pretanh_rms": rms(pre_tanh),
                "generator_pretanh_absmax": absmax(pre_tanh),
                "waveform_rms": rms(recon),
                "waveform_saturation_fraction": (recon.detach().abs() > 0.95).float().mean().item(),
            }
            log_records.append(record)
            log_path.write_text(json.dumps({"ablation": label, "steps": args.steps, "log": log_records}, indent=2))

            print(
                f"step {step:>5d}  total {loss.item():.4f}  grad_norm {grad_norm.item():>9.3f}  best {best_loss:.4f}  "
                f"g_dec {gate_decoder.item():.4f}  to_out_norm {record['to_output_weight_norm']:.3f}  "
                f"skip_rms {record['skip_rms']:.4f}  temp_raw_rms {record['temporal_raw_rms']:.4f}  "
                f"temp_gated_rms {record['temporal_gated_rms']:.4f}  gen_in_rms {record['generator_in_rms']:.4f}  "
                f"gen_pretanh_absmax {record['generator_pretanh_absmax']:.4f}  wav_sat_frac {record['waveform_saturation_fraction']:.4f}"
            )

    torchaudio.save(str(out_dir / "recon_final.wav"), recon[0].detach().cpu(), SAMPLE_RATE)
    print(f"samples written to: {out_dir}")
    print(f"JSON log written to: {log_path}")

    max_grad_norm = max(r["grad_norm"] for r in log_records)
    return {
        "ablation": label,
        "best_loss": best_loss,
        "final_loss": log_records[-1]["total_loss"],
        "max_grad_norm": max_grad_norm,
        "final_generator_in_rms": log_records[-1]["generator_in_rms"],
        "final_waveform_saturation_fraction": log_records[-1]["waveform_saturation_fraction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose generator_in and ablate decoder-v3's output gain path")
    parser.add_argument("--ablation", choices=[*ALL_ABLATIONS, "all"], default="all")
    parser.add_argument(
        "--gate-branch",
        choices=["temporal", "skip"],
        default="temporal",
        help="which branch gate_decoder scales via ContinuousDecoder.fuse (fuse(base, x, gate) = "
        "base + gate*x). 'temporal' (default) is this script's original ablation design -- "
        "gate_decoder=0 means 'only skip(z) reaches the generator' (see 'no_temporal' above). "
        "'skip' matches production's decoder.step instead (skip_gate scales skip, temporal is "
        "full-weight) -- the two are NOT equivalent at the same gate value; pass 'skip' to "
        "directly compare against what decoder.step actually does.",
    )
    parser.add_argument(
        "--gate-values",
        type=float,
        nargs="+",
        default=None,
        help="run a fixed-gate sweep instead of --ablation: one run per value, "
        "each with the gate fixed at that value (freeze_gate mechanics) and excluded from the optimizer. "
        "e.g. --gate-values 0.05 0.10 0.15 0.20",
    )
    parser.add_argument("--fixed-gain-target", type=float, default=1.0, help="target RMS for --ablation fixed_gain")
    parser.add_argument(
        "--no-decoder-self-recurrence",
        action="store_true",
        help="drop each decoder cell's own h_{t-1} from its candidate's input (cross-timescale inputs are unaffected); "
        "tests whether decoder's self-recurrence contributes to the intermittent spikes seen with saturated hidden states (Run 27)",
    )
    parser.add_argument(
        "--decoder-fixed-tau",
        action="store_true",
        help="replace each decoder branch's learned adaptive tau with a constant (that branch's own tau_range midpoint), "
        "removing the h_{t-1} -> tau_t -> alpha_t feedback path; mirrors Run 15's encoder test, applied to the decoder",
    )
    parser.add_argument(
        "--disable-decoder-cross-connections",
        action="store_true",
        help="remove cross-timescale connections between decoder's fast/mid/slow branches (each cell only sees its own state); "
        "tests whether saturated cross-branch signals amplify instability (Run 27)",
    )
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--gate-init", type=float, default=0.1)
    parser.add_argument("--decoder-gate-init", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--alignment-delay",
        type=int,
        default=None,
        help="see train_stage1.py --alignment-delay. Default: generator.total_stride (240).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.gate_values:
        summaries = [run_ablation(args, "gate_sweep", gate_override=g) for g in args.gate_values]
    else:
        ablations = ALL_ABLATIONS if args.ablation == "all" else [args.ablation]
        summaries = [run_ablation(args, ablation) for ablation in ablations]

    if len(summaries) > 1:
        print("\n=== summary ===")
        header = f"{'ablation':>12}  {'best_loss':>10}  {'final_loss':>10}  {'max_grad_norm':>13}  {'final_gen_in_rms':>16}  {'final_sat_frac':>14}"
        print(header)
        for s in summaries:
            print(
                f"{s['ablation']:>12}  {s['best_loss']:>10.4f}  {s['final_loss']:>10.4f}  "
                f"{s['max_grad_norm']:>13.3f}  {s['final_generator_in_rms']:>16.4f}  {s['final_waveform_saturation_fraction']:>14.4f}"
            )


if __name__ == "__main__":
    main()
