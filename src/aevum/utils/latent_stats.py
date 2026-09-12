"""Diagnostic statistics for a latent sequence, used to compare encoder z(t)
against other latent distributions the decoder is known to train stably on
(docs/reports/stage1_v0.md, Run 25: encoder-decoder interface diagnosis).
"""

from __future__ import annotations

import torch


def latent_stats(name: str, z: torch.Tensor, eps: float = 1e-8) -> None:
    """Print shape/scale/temporal-smoothness statistics for ``z``: ``[B, T, C]``."""
    z = z.detach().float()
    delta = z[:, 1:] - z[:, :-1]

    rms = z.pow(2).mean().sqrt()
    abs_max = z.abs().max()

    ch_std = z.std(dim=(0, 1))
    ch_mean = z.mean(dim=(0, 1))

    norms = z.norm(dim=-1)

    z1 = torch.nn.functional.normalize(z[:, :-1], dim=-1, eps=eps)
    z2 = torch.nn.functional.normalize(z[:, 1:], dim=-1, eps=eps)
    temporal_cos = (z1 * z2).sum(dim=-1)

    print(f"\n[{name}]")
    print("shape:", tuple(z.shape))
    print("mean:", z.mean().item())
    print("std:", z.std().item())
    print("rms:", rms.item())
    print("abs_max:", abs_max.item())

    print("channel mean abs:", ch_mean.abs().mean().item())
    print("channel std min:", ch_std.min().item())
    print("channel std median:", ch_std.median().item())
    print("channel std max:", ch_std.max().item())

    print("delta rms:", delta.pow(2).mean().sqrt().item())
    print("delta abs max:", delta.abs().max().item())

    print("temporal cosine mean:", temporal_cos.mean().item())

    print("norm/t mean:", norms.mean().item())
    print("norm/t std:", norms.std().item())
    print("norm/t min:", norms.min().item())
    print("norm/t max:", norms.max().item())
