#!/usr/bin/env python3
"""Entry point for evaluating a trained AEVUM codec checkpoint.

Usage:
    uv run scripts/evaluate.py --checkpoint outputs/model.pt --config configs/base.yaml
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AEVUM codec checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to an evaluation config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Evaluation pipeline not implemented yet (checkpoint={args.checkpoint}, config={args.config})"
    )


if __name__ == "__main__":
    main()
