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
DataLoader uses pinned memory and persistent workers.

--compile and --amp exist as opt-in flags but neither has actually helped
on Stage 1's shape (encoder/decoder: 200+200 sequential steps of small
192-384-dim ops) -- both target compute throughput, and this loop isn't
compute bound:
- --compile (mode='reduce-overhead', CUDA graphs): dynamo fully unrolls
  the Python for-loop into one huge graph before it can capture anything,
  so first-call compilation can hang for a very long time (observed: still
  compiling after 10+ minutes). Would need per-step compilation
  (compiling encoder.step/decoder.step individually, called from the
  Python loop) to be worth trying again -- not implemented.
- --amp (bf16 autocast): measured SLOWER end-to-end. Autocast inserts
  fp32<->bf16 cast kernels around each eligible op, which adds kernel
  launches in a loop that was already launch-overhead bound instead of
  compute bound -- the ops are too small for bf16 tensor-core throughput
  to matter, so this is pure added overhead. (Also: torch.stft, used by
  the mel/STFT loss terms, goes through cuFFT and does not support bf16
  at all -- the loss is computed in fp32 regardless of --amp.)

Usage:
    uv run scripts/train_stage1.py --data-root data/raw --librispeech-url dev-clean --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

from aevum.data.librispeech import LibriSpeechSegments
from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.training.losses.reconstruction import ReconstructionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AEVUM Stage 1 dense continuous autoencoder")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="dev-clean")
    parser.add_argument(
        "--val-url",
        type=str,
        default="test-clean",
        help="LibriSpeech split used for validation. Must differ from --librispeech-url -- LibriSpeech "
        "splits have disjoint speakers by construction, so this alone gives held-out (unseen-speaker) "
        "validation instead of scoring a training example (see docs/reports/stage1_v0.md). Default "
        "pairs with --librispeech-url's default of dev-clean without colliding.",
    )
    parser.add_argument("--val-data-root", type=str, default=None, help="defaults to --data-root")
    parser.add_argument(
        "--val-max-segments",
        type=int,
        default=256,
        help="cap the number of validation segments used per epoch for the scheduler/best-checkpoint "
        "loss (0 = use the full split). Aggregated, not a single clip -- see docs/reports/stage1_v0.md.",
    )
    parser.add_argument(
        "--val-panel-size",
        type=int,
        default=4,
        help="number of fixed validation clips saved as audio each epoch for listening",
    )
    parser.add_argument(
        "--alignment-delay",
        type=int,
        default=None,
        help="samples of delay between output and target in the reconstruction loss (see "
        "align_for_reconstruction). Default: the model's own total_stride (240 = one 10ms "
        "frontend frame), the minimum needed for a causally fair comparison. Pass 0 to "
        "reproduce the old (misaligned) loss for an A/B comparison.",
    )
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a full training checkpoint (.pt, as written by this script) to resume from: "
        "model + optimizer + scheduler + epoch + best_val_loss + RNG state. Continues the same "
        "experiment (same output dir/log recommended) rather than starting a new one.",
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="path to a checkpoint (.pt) to load only model weights from -- fresh optimizer/scheduler/"
        "epoch/best_val_loss. For fine-tuning or starting a new experiment from a pretrained model, "
        "not for continuing an interrupted run (use --resume for that). Accepts either this script's "
        "full checkpoint format or a bare state_dict.",
    )
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
        help="bf16 autocast on forward+loss. Measured SLOWER on Stage 1 (docs/reports/stage1_v0.md): "
        "the encoder/decoder loop is launch-overhead bound with small (192-384-dim) ops, so autocast's "
        "per-op fp32<->bf16 cast kernels add launches rather than speeding up compute that was never "
        "the bottleneck. Not recommended for this model shape; kept for future stages with bigger ops.",
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

    if args.val_url == args.librispeech_url:
        raise ValueError(
            f"--val-url ({args.val_url!r}) equals --librispeech-url: validation would score "
            "examples the model can also train on. Use a disjoint LibriSpeech split, e.g. "
            "dev-clean when training on train-clean-100."
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

    # Held-out split (disjoint speakers from --librispeech-url by LibriSpeech's own split
    # construction) -- val_loss below is an aggregate over this, not one memorizable clip.
    val_dataset = LibriSpeechSegments(
        root=args.val_data_root or args.data_root, url=args.val_url, segment_seconds=args.segment_seconds
    )
    val_indices = list(range(len(val_dataset)))
    if args.val_max_segments and len(val_indices) > args.val_max_segments:
        g = torch.Generator().manual_seed(0)
        val_indices = torch.randperm(len(val_indices), generator=g)[: args.val_max_segments].tolist()
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Fixed panel of held-out clips for periodic listening -- reconstruction quality on
    # varying training batches is too noisy step-to-step to judge progress by ear.
    panel_size = min(args.val_panel_size, len(val_dataset))
    val_panel = torch.stack([val_dataset[i] for i in range(panel_size)]).to(device)

    model = DenseContinuousAutoencoder().to(device)
    start_epoch = 0
    best_val_loss = float("inf")
    optimizer_state = None
    scheduler_state = None
    rng_state = None
    resumed_epoch_log: list[dict] = []

    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"initialized model weights from {args.init_weights} (fresh optimizer/scheduler/epoch)", flush=True)
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if not (isinstance(ckpt, dict) and "model" in ckpt):
            raise ValueError(
                f"{args.resume} is not a full training checkpoint (no optimizer/scheduler/epoch/RNG "
                "state) -- use --init-weights to load it as a fresh run's starting weights instead."
            )
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        optimizer_state = ckpt["optimizer"]
        scheduler_state = ckpt.get("scheduler")
        rng_state = ckpt.get("rng_state")
        existing_log = Path(args.log_file) if args.log_file else Path(args.checkpoint_dir) / "train_log.json"
        if existing_log.exists():
            resumed_epoch_log = json.loads(existing_log.read_text()).get("epochs", [])
        print(
            f"resumed full training state from {args.resume}: epoch {start_epoch}, "
            f"best_val_loss {best_val_loss:.4f}",
            flush=True,
        )
        if rng_state is not None:
            torch.set_rng_state(rng_state["torch"])
            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])

    alignment_delay = model.total_stride if args.alignment_delay is None else args.alignment_delay
    criterion = ReconstructionLoss(
        wav_weight=args.wav_weight,
        mel_weight=args.mel_weight,
        stft_weight=args.stft_weight,
        delay=alignment_delay,
    ).to(device)
    # Fused AdamW: one CUDA kernel across all parameter tensors instead of one per tensor --
    # this model is many small Linear layers, so unfused Adam issues many tiny kernel launches
    # per optimizer.step(), same launch-overhead problem as the encoder/decoder loop.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(device.type == "cuda"))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

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
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    def save_checkpoint(path: Path) -> None:
        """Full training state, not just weights -- see --resume/--init-weights above."""
        torch.save(
            {
                "model": checkpoint_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            path,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i in range(val_panel.shape[0]):
        torchaudio.save(str(samples_dir / f"val_target_{i}.wav"), val_panel[i].cpu(), 24_000)

    log_path = Path(args.log_file) if args.log_file else checkpoint_dir / "train_log.json"
    run_config = vars(args) | {
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "steps_per_epoch": steps_per_epoch,
        "val_segments_used": len(val_indices),
        "alignment_delay_samples": alignment_delay,
    }
    epoch_log: list[dict] = resumed_epoch_log

    def write_log() -> None:
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"config": run_config, "epochs": epoch_log}, indent=2))
        tmp.replace(log_path)

    start_time = time.perf_counter()
    audio_seconds_seen = sum(r.get("audio_seconds_seen", 0.0) for r in resumed_epoch_log[-1:])

    for epoch in range(start_epoch, args.epochs):
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
            # Loss is always computed in fp32, outside autocast: torch.stft (used by both the
            # mel and multi-res STFT terms) goes through cuFFT, which does not support bf16 at
            # all and errors instead of upcasting -- autocast doesn't cover this op, so a bf16
            # `reconstructed` has to be cast back explicitly rather than left to autocast.
            losses = criterion(waveform, reconstructed.float())

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
        val_running = torch.zeros((), device=device)
        val_batches = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            for val_waveform in val_loader:
                val_waveform = val_waveform.to(device, non_blocking=True)
                val_out = model(val_waveform)
                val_running += criterion(val_waveform, val_out.float())["total"].detach()
                val_batches += 1
            val_loss = (val_running / val_batches).item()

            panel_recon = model(val_panel)  # fixed panel, saved for listening only
        for i in range(val_panel.shape[0]):
            torchaudio.save(str(samples_dir / f"val_epoch{epoch}_{i}.wav"), panel_recon[i].float().cpu(), 24_000)

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
            for i in range(val_panel.shape[0]):
                torchaudio.save(str(samples_dir / f"val_best_{i}.wav"), panel_recon[i].float().cpu(), 24_000)
            save_checkpoint(checkpoint_dir / "stage1_best.pt")

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
            save_checkpoint(checkpoint_dir / f"stage1_epoch{epoch}.pt")

    save_checkpoint(checkpoint_dir / "stage1_final.pt")
    write_log()
    print(f"\ntrain log written to: {log_path}", flush=True)


if __name__ == "__main__":
    main()
