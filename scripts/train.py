#!/usr/bin/env python3
"""Entry point for training AEVUM codec models.

Usage:
    uv run scripts/train.py --config configs/base.yaml
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AEVUM codec model")
    parser.add_argument("--config", type=str, required=True, help="Path to a training config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Training pipeline not implemented yet (config={args.config})")


if __name__ == "__main__":
    main()
