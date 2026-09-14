#!/usr/bin/env python3
"""Download/extract/index a LibriSpeech split, separately from training.

Runs the same one-time preparation LibriSpeechSegments does internally
(download, extract, file-list scan, segment index build), each stage
printing its own progress, so it can be watched to completion on its own
before starting a training run.

Usage:
    uv run scripts/prepare_data.py --data-root data/raw --librispeech-url train-clean-100
"""

import argparse

from aevum.data.librispeech import LibriSpeechSegments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a LibriSpeech split for training")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--librispeech-url", type=str, default="train-clean-100")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LibriSpeechSegments(
        root=args.data_root, url=args.librispeech_url, segment_seconds=args.segment_seconds
    )
    print(f"dataset ready: {len(dataset)} segments", flush=True)


if __name__ == "__main__":
    main()
