# AEVUM

**AEVUM** (Latin: *aevum* — "the flow of time / duration") is a continuous-time, event-driven neural speech codec developed by [Manifestro](https://github.com/manifestro).

## About

Unlike conventional neural codecs, which tokenize audio at a fixed frame rate, AEVUM represents speech as a stream of irregular, information-triggered events. The codec continuously observes the input signal and maintains an internal continuous-time state; a new discrete packet is only emitted when the current state diverges enough from what the decoder can already predict on its own. As a result, both **time** (when to transmit) and **bitrate** (how much to transmit) become adaptive to the local complexity of the speech signal, instead of being fixed.

Initial target operating point: 24 kHz mono input, causal/streaming with zero lookahead, ~8-20 events/sec, ~0.5-1.5 kbps average bitrate.

The full architecture, math, training curriculum, and metrics are specified in [`docs/tech_spec.md`](./docs/tech_spec.md). This repository holds the model implementation, training/evaluation pipelines, and supporting tooling for the project.

## Project layout

```
aevum/
├── src/aevum/          # Main package
│   ├── models/         # Encoder / decoder / entropy model architectures
│   ├── data/           # Datasets and data loading utilities
│   ├── training/       # Training loops, losses, schedulers
│   ├── inference/       # Encode/decode inference pipeline
│   └── utils/           # Shared helpers (logging, config, metrics)
├── scripts/             # CLI entry points (train.py, evaluate.py)
├── configs/             # Experiment configuration files
├── data/                # Local datasets (gitignored, raw/ + processed/)
├── notebooks/            # Exploratory notebooks
├── tests/                # Unit tests
├── docs/                 # Additional documentation
├── outputs/               # Model checkpoints & artifacts (gitignored)
└── logs/                  # Run logs (gitignored)
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run the test suite
uv run pytest

# Lint
uv run ruff check .

# Train (once the pipeline is implemented)
uv run scripts/train.py --config configs/base.yaml
```

Python >= 3.11 is required.

## Status

Early scaffolding stage. The architecture and training curriculum are specified in [`docs/tech_spec.md`](./docs/tech_spec.md); implementation has not started yet.

## License

Proprietary — © Manifestro. All rights reserved.
