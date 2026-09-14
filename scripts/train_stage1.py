#!/usr/bin/env python3
"""Stage 1 training: dense continuous autoencoder, no quantization/predictor/event gate.

Validates the architectural hypothesis in isolation (tech_spec.md section 41,
Stage 1) before any codec-specific machinery is added on top. Uses the
architecture validated end-to-end in docs/reports/stage1_v0.md (Runs 16-31):
encoder direct-residual + gated fusion, decoder self-recurrence removed +
fixed-gate skip.

Trains epoch-by-epoch (one full pass over the dataset per epoch) rather than
by raw step count -- per-step loss on diverse real batches is too noisy to
read directly, per-epoch mean loss is not. Each epoch shows a tqdm progress
bar and ends with one logged summary line (train means + val loss), appended
to a JSON log file (default outputs/train_log.json) as well as printed.

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
from tqdm import tqdm

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
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs")
    parser.add_argument("--log-file", type=str, default=None, help="JSON log path, default <checkpoint-dir>/train_log.json")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint (.pt) to resume model weights from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = LibriSpeechSegments(
        root=args.data_root, url=args.librispeech_url, segment_seconds=args.segment_seconds
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    steps_per_epoch = len(loader)

    # Fixed held-out clip for periodic listening/eval — reconstruction quality on
    # varying training batches is too noisy step-to-step to judge progress by ear.
    torch.manual_seed(0)
    val_waveform = dataset[0].unsqueeze(0).to(device)

    model = DenseContinuousAutoencoder().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed model weights from {args.resume}")
    criterion = ReconstructionLoss(
        wav_weight=args.wav_weight, mel_weight=args.mel_weight, stft_weight=args.stft_weight
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
        running = {"total": 0.0, "wav": 0.0, "mel": 0.0, "stft": 0.0, "grad_norm": 0.0}
        progress = tqdm(loader, desc=f"epoch {epoch:>4d}", unit="batch")
        for waveform in progress:
            waveform = waveform.to(device)

            reconstructed = model(waveform)
            losses = criterion(waveform, reconstructed)

            optimizer.zero_grad()
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            current_lr = optimizer.param_groups[0]["lr"]  # LR actually used for this step, before any scheduler advance
            optimizer.step()
            if args.lr_scheduler == "cosine":
                scheduler.step()

            audio_seconds_seen += waveform.shape[0] * args.segment_seconds
            running["total"] += losses["total"].item()
            running["wav"] += losses["wav"].item()
            running["mel"] += losses["mel"].item()
            running["stft"] += losses["stft"].item()
            running["grad_norm"] += grad_norm.item()
            progress.set_postfix(loss=f"{losses['total'].item():.3f}", lr=f"{current_lr:.1e}")

        mean = {k: v / steps_per_epoch for k, v in running.items()}
        elapsed = time.perf_counter() - start_time

        gates = {
            "fast": model.encoder.fast_gate.item(),
            "mid": model.encoder.mid_gate.item(),
            "slow": model.encoder.slow_gate.item(),
        }

        model.eval()
        with torch.no_grad():
            val_recon = model(val_waveform)
            val_loss = criterion(val_waveform, val_recon)["total"].item()

        torchaudio.save(str(samples_dir / f"val_epoch{epoch}.wav"), val_recon[0].cpu(), 24_000)

        if args.lr_scheduler == "plateau":
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  [lr] val loss stalled for {args.lr_patience} epochs -- cutting lr {lr_before:.2e} -> {lr_after:.2e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torchaudio.save(str(samples_dir / "val_best.wav"), val_recon[0].cpu(), 24_000)
            torch.save(model.state_dict(), checkpoint_dir / "stage1_best.pt")

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
            f"val {val_loss:.4f} (best {best_val_loss:.4f}){'  *' if is_best else ''}  elapsed {elapsed:.0f}s"
        )

        if (epoch + 1) % args.checkpoint_every_epochs == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"stage1_epoch{epoch}.pt")

    torch.save(model.state_dict(), checkpoint_dir / "stage1_final.pt")
    write_log()
    print(f"\ntrain log written to: {log_path}")


if __name__ == "__main__":
    main()
