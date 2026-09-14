#!/usr/bin/env python3
"""Stage 1 training: dense continuous autoencoder, no quantization/predictor/event gate.

Validates the architectural hypothesis in isolation (tech_spec.md section 41,
Stage 1) before any codec-specific machinery is added on top. Uses the
architecture validated end-to-end in docs/reports/stage1_v0.md (Runs 16-31):
encoder direct-residual + gated fusion, decoder self-recurrence removed +
fixed-gate skip.

Trains epoch-by-epoch (one full pass over the dataset per epoch) rather than
by raw step count -- per-step loss on diverse real batches is too noisy to
read directly, per-epoch mean loss is not. Each epoch prints periodic
in-progress batch lines (plain prints, not tqdm -- tqdm's carriage-return
redraw does not render through `!uv run ...` in Colab, see
docs/reports/stage1_v0.md) and ends with one logged summary line (train
means + val loss), appended to a JSON log file (default
outputs/train_log.json) as well as printed.

Speed: the encoder/decoder are sequential 200-step Python loops per batch,
which docs/reports/stage1_v0.md's throughput benchmark identified as
kernel-launch-overhead bound, not compute bound. Always-on fixes (no
numerical/behavioral change): running loss/grad-norm stats are accumulated
as GPU tensors and only synced (`.item()`) at --log-every cadence instead
of every single batch, since a CPU-GPU sync every step serializes exactly
the launch-bound loop this is trying to speed up; TF32 matmul + cudnn
autotune are enabled; the optimizer uses the fused CUDA AdamW kernel; the
DataLoader uses pinned memory and persistent workers. Opt-in, higher-payoff
but numerically-sensitive: --compile (torch.compile, mode='reduce-overhead',
uses CUDA graphs internally -- attacks the launch-overhead bottleneck
directly) and --amp (bf16 autocast on forward+loss, big matmul speedup on
A100 tensor cores, no GradScaler needed since bf16's exponent range matches
fp32). Verify a short run's loss trajectory against an unflagged run before
trusting a long one with either flag on.

Usage:
    uv run scripts/train_stage1.py --data-root data/raw --librispeech-url dev-clean --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AEVUM Stage 1 dense continuous autoencoder")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau", "cosine"],
        default="plateau",
        help="'plateau' (default): cut LR when val loss stalls/regresses across epochs. "
        "'cosine': smooth decay from --lr to --lr-min over the full --epochs run.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5, help="plateau: multiply LR by this when triggered")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="plateau: number of epochs with no val improvement before cutting LR",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="floor for both 'plateau' and 'cosine'")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--wav-weight",
        type=float,
        default=1.0,
        help="weight on the raw-waveform L1 term. Its natural magnitude (~0.04) is far smaller than "
        "mel/stft (~0.5-0.8), so at the default 1.0 it barely contributes to the gradient and stays "
        "essentially unoptimized on diverse real data (see docs/reports/stage1_v0.md) -- try 10-20 to "
        "make its influence comparable to mel/stft.",
    )
    parser.add_argument("--mel-weight", type=float, default=1.0)
    parser.add_argument("--stft-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--log-every", type=int, default=20, help="print an in-progress line every N batches within an epoch"
    )
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs")
    parser.add_argument("--log-file", type=str, default=None, help="JSON log path, default <checkpoint-dir>/train_log.json")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint (.pt) to resume model weights from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(model, mode='reduce-overhead') -- uses CUDA graphs to attack the "
        "kernel-launch-overhead bottleneck in the encoder/decoder's per-step loop. Opt-in: "
        "verify a short run's loss trajectory matches an uncompiled run before trusting a long one.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on forward+loss (no GradScaler needed). Opt-in: verify a short run's "
        "loss trajectory before trusting a long one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # TF32 matmul (Linear layers) + cudnn autotune (Conv1d in frontend/generator) -- free on
    # A100, no numerical-precision opt-in required (TF32 is not full mixed precision, unlike
    # --amp). Batch/segment shape is constant per epoch (drop_last=True) so autotune pays off
    # instead of re-tuning every batch.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    print(
        f"starting run: device={device} data_root={args.data_root} url={args.librispeech_url} "
        f"compile={args.compile} amp_bf16={args.amp} num_workers={args.num_workers}",
        flush=True,
    )

    dataset = LibriSpeechSegments(
        root=args.data_root, url=args.librispeech_url, segment_seconds=args.segment_seconds
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = len(loader)

    # Fixed held-out clip for periodic listening/eval — reconstruction quality on
    # varying training batches is too noisy step-to-step to judge progress by ear.
    torch.manual_seed(0)
    val_waveform = dataset[0].unsqueeze(0).to(device)

    model = DenseContinuousAutoencoder().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed model weights from {args.resume}", flush=True)
    criterion = ReconstructionLoss(
        wav_weight=args.wav_weight, mel_weight=args.mel_weight, stft_weight=args.stft_weight
    ).to(device)
    # Fused AdamW: one CUDA kernel across all parameter tensors instead of one per tensor --
    # this model is many small Linear layers, so unfused Adam issues many tiny kernel launches
    # per optimizer.step(), same launch-overhead problem as the encoder/decoder loop.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(device.type == "cuda"))

    # Keep a handle to the uncompiled module for checkpointing: torch.compile's wrapper can
    # save/load state_dict keys with a "_orig_mod." prefix depending on version, which would
    # break resuming a compiled run's checkpoint into a plain (--compile off) run later.
    # Same underlying parameters either way -- optimizer.step() updates both views.
    checkpoint_model = model
    if args.compile:
        # mode='reduce-overhead' uses CUDA graphs internally -- directly targets the
        # kernel-launch-overhead bottleneck in the per-step recurrence (see module docstring).
        model = torch.compile(model, mode="reduce-overhead")

    scheduler = None
    if args.lr_scheduler == "plateau":
        # Reacts to val loss stalling/regressing across epochs (see
        # docs/reports/stage1_v0.md for the real-LibriSpeech runs that
        # motivated this). Cuts LR instead of raising it.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr_min
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(samples_dir / "val_target.wav"), val_waveform[0].cpu(), 24_000)

    log_path = Path(args.log_file) if args.log_file else checkpoint_dir / "train_log.json"
    run_config = vars(args) | {
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "steps_per_epoch": steps_per_epoch,
    }
    epoch_log: list[dict] = []

    def write_log() -> None:
        log_path.write_text(json.dumps({"config": run_config, "epochs": epoch_log}, indent=2))

    best_val_loss = float("inf")
    start_time = time.perf_counter()
    audio_seconds_seen = 0.0

    for epoch in range(args.epochs):
        model.train()
        # Accumulated as GPU tensors, not Python floats: a per-step `.item()` forces a
        # CPU-GPU sync, which serializes exactly the launch-overhead-bound loop this whole
        # file is trying to speed up. Only synced at --log-every cadence and epoch end.
        running = {k: torch.zeros((), device=device) for k in ("total", "wav", "mel", "stft", "grad_norm")}
        epoch_start = time.perf_counter()
        for batch_idx, waveform in enumerate(loader, 1):
            waveform = waveform.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                reconstructed = model(waveform)
                losses = criterion(waveform, reconstructed)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            current_lr = optimizer.param_groups[0]["lr"]  # LR actually used for this step, before any scheduler advance
            optimizer.step()
            if args.lr_scheduler == "cosine":
                scheduler.step()

            audio_seconds_seen += waveform.shape[0] * args.segment_seconds
            running["total"] += losses["total"].detach()
            running["wav"] += losses["wav"].detach()
            running["mel"] += losses["mel"].detach()
            running["stft"] += losses["stft"].detach()
            running["grad_norm"] += grad_norm.detach()

            if batch_idx % args.log_every == 0 or batch_idx == steps_per_epoch:
                batch_elapsed = time.perf_counter() - epoch_start
                print(
                    f"  epoch {epoch:>4d} batch {batch_idx:>4d}/{steps_per_epoch}  "
                    f"loss {losses['total'].item():.4f}  lr {current_lr:.2e}  {batch_elapsed:.0f}s",
                    flush=True,
                )

        mean = {k: (v / steps_per_epoch).item() for k, v in running.items()}
        elapsed = time.perf_counter() - start_time

        gates = {
            "fast": model.encoder.fast_gate.item(),
            "mid": model.encoder.mid_gate.item(),
            "slow": model.encoder.slow_gate.item(),
        }

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            val_recon = model(val_waveform)
            val_loss = criterion(val_waveform, val_recon)["total"].item()

        torchaudio.save(str(samples_dir / f"val_epoch{epoch}.wav"), val_recon[0].float().cpu(), 24_000)

        if args.lr_scheduler == "plateau":
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(
                    f"  [lr] val loss stalled for {args.lr_patience} epochs -- cutting lr {lr_before:.2e} -> {lr_after:.2e}",
                    flush=True,
                )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torchaudio.save(str(samples_dir / "val_best.wav"), val_recon[0].float().cpu(), 24_000)
            torch.save(checkpoint_model.state_dict(), checkpoint_dir / "stage1_best.pt")

        record = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "audio_seconds_seen": audio_seconds_seen,
            "lr": current_lr,
            "total_loss": mean["total"],
            "wav_loss": mean["wav"],
            "mel_loss": mean["mel"],
            "stft_loss": mean["stft"],
            "grad_norm": mean["grad_norm"],
            "gate_fast": gates["fast"],
            "gate_mid": gates["mid"],
            "gate_slow": gates["slow"],
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        epoch_log.append(record)
        write_log()
        print(
            f"epoch {epoch:>4d}  total {mean['total']:.4f}  wav {mean['wav']:.4f}  "
            f"mel {mean['mel']:.4f}  stft {mean['stft']:.4f}  grad_norm {mean['grad_norm']:.3f}  "
            f"gates(f/m/s) {gates['fast']:.3f}/{gates['mid']:.3f}/{gates['slow']:.3f}  lr {current_lr:.2e}  "
            f"val {val_loss:.4f} (best {best_val_loss:.4f}){'  *' if is_best else ''}  elapsed {elapsed:.0f}s",
            flush=True,
        )

        if (epoch + 1) % args.checkpoint_every_epochs == 0:
            torch.save(checkpoint_model.state_dict(), checkpoint_dir / f"stage1_epoch{epoch}.pt")

    torch.save(checkpoint_model.state_dict(), checkpoint_dir / "stage1_final.pt")
    write_log()
    print(f"\ntrain log written to: {log_path}", flush=True)


if __name__ == "__main__":
    main()
