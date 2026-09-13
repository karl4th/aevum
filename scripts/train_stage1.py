#!/usr/bin/env python3
"""Stage 1 training: dense continuous autoencoder, no quantization/predictor/event gate.

Validates the architectural hypothesis in isolation (tech_spec.md section 41,
Stage 1) before any codec-specific machinery is added on top. Uses the
architecture validated end-to-end in docs/reports/stage1_v0.md (Runs 16-31):
encoder direct-residual + gated fusion, decoder self-recurrence removed +
fixed-gate skip.

Every logged step (train and validation) is appended to a JSON log file
(default outputs/train_log.json) as well as printed, so a run's full
history can be inspected/plotted afterward rather than only skimmed from
console output.

Usage:
    uv run scripts/train_stage1.py --data-root data/raw --librispeech-url dev-clean --steps 5000
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
        help="'plateau' (default): cut LR when val loss stalls/regresses -- reactive, doesn't need --steps guessed "
        "correctly in advance. 'cosine': smooth decay from --lr to --lr-min over the full --steps run.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5, help="plateau: multiply LR by this when triggered")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="plateau: number of val checks (every --sample-every steps) with no improvement before cutting LR",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="floor for both 'plateau' and 'cosine'")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
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

    # Fixed held-out clip for periodic listening/eval — reconstruction quality on
    # varying training batches is too noisy step-to-step to judge progress by ear.
    torch.manual_seed(0)
    val_waveform = dataset[0].unsqueeze(0).to(device)

    model = DenseContinuousAutoencoder().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed model weights from {args.resume}")
    criterion = ReconstructionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if args.lr_scheduler == "plateau":
        # Reacts to val loss stalling/regressing (exactly the symptom that
        # motivated this: docs/reports/stage1_v0.md, real-LibriSpeech run --
        # loss oscillating in a band without net progress, val loss ticking
        # up between checks). Cuts LR instead of raising it: the oscillation
        # pattern (bouncing, not a slow monotonic crawl) indicates the LR is
        # too large for the local loss landscape once early coarse progress
        # is exhausted, not too small.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.lr_min
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr_min)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(samples_dir / "val_target.wav"), val_waveform[0].cpu(), 24_000)

    log_path = Path(args.log_file) if args.log_file else checkpoint_dir / "train_log.json"
    run_config = vars(args) | {
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    train_log: list[dict] = []
    val_log: list[dict] = []

    def write_log() -> None:
        log_path.write_text(json.dumps({"config": run_config, "train": train_log, "val": val_log}, indent=2))

    best_val_loss = float("inf")
    step = 0
    start_time = time.perf_counter()
    audio_seconds_seen = 0.0
    while step < args.steps:
        for waveform in loader:
            waveform = waveform.to(device)

            reconstructed = model(waveform)
            losses = criterion(waveform, reconstructed)

            optimizer.zero_grad()
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            current_lr = optimizer.param_groups[0]["lr"]  # the LR actually used for this step's update, before any scheduler advance
            optimizer.step()
            if args.lr_scheduler == "cosine":
                scheduler.step()

            audio_seconds_seen += waveform.shape[0] * args.segment_seconds

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                gates = {
                    "fast": model.encoder.fast_gate.item(),
                    "mid": model.encoder.mid_gate.item(),
                    "slow": model.encoder.slow_gate.item(),
                }
                record = {
                    "step": step,
                    "elapsed_sec": elapsed,
                    "audio_seconds_seen": audio_seconds_seen,
                    "lr": current_lr,
                    "total_loss": losses["total"].item(),
                    "wav_loss": losses["wav"].item(),
                    "mel_loss": losses["mel"].item(),
                    "stft_loss": losses["stft"].item(),
                    "grad_norm": grad_norm.item(),
                    "gate_fast": gates["fast"],
                    "gate_mid": gates["mid"],
                    "gate_slow": gates["slow"],
                }
                train_log.append(record)
                write_log()
                print(
                    f"step {step:>7d}  total {losses['total'].item():.4f}  "
                    f"wav {losses['wav'].item():.4f}  mel {losses['mel'].item():.4f}  "
                    f"stft {losses['stft'].item():.4f}  grad_norm {grad_norm.item():.3f}  "
                    f"gates(f/m/s) {gates['fast']:.3f}/{gates['mid']:.3f}/{gates['slow']:.3f}  "
                    f"lr {current_lr:.2e}  elapsed {elapsed:.0f}s"
                )

            if step % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    val_recon = model(val_waveform)
                    val_loss = criterion(val_waveform, val_recon)["total"].item()
                model.train()

                torchaudio.save(str(samples_dir / f"val_step{step}.wav"), val_recon[0].cpu(), 24_000)
                val_log.append(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "best_val_loss": min(best_val_loss, val_loss),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                write_log()
                print(f"  [val] step {step:>7d}  loss {val_loss:.4f}  (best {best_val_loss:.4f})")

                if args.lr_scheduler == "plateau":
                    lr_before = optimizer.param_groups[0]["lr"]
                    scheduler.step(val_loss)
                    lr_after = optimizer.param_groups[0]["lr"]
                    if lr_after < lr_before:
                        print(f"  [lr] val loss stalled for {args.lr_patience} checks -- cutting lr {lr_before:.2e} -> {lr_after:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torchaudio.save(str(samples_dir / "val_best.wav"), val_recon[0].cpu(), 24_000)
                    torch.save(model.state_dict(), checkpoint_dir / "stage1_best.pt")

            if step % args.checkpoint_every == 0 and step > 0:
                torch.save(model.state_dict(), checkpoint_dir / f"stage1_step{step}.pt")

            step += 1
            if step >= args.steps:
                break

    torch.save(model.state_dict(), checkpoint_dir / "stage1_final.pt")
    write_log()
    print(f"\ntrain log written to: {log_path}")


if __name__ == "__main__":
    main()
