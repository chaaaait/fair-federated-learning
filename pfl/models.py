"""Models and flat-parameter helpers.

GroupNorm instead of BatchNorm: BatchNorm mixes statistics across the examples of a batch, so one
example's gradient depends on the others and per-example clipping no longer bounds sensitivity.
Every model exposes `body` (shared representation) and `head` (last layer). FedPer keeps `head`
on the client.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _cgr(cin: int, cout: int, groups: int) -> list:
    return [nn.Conv2d(cin, cout, kernel_size=3, padding=1), nn.GroupNorm(groups, cout), nn.ReLU()]


class CNN(nn.Module):
    """5 conv layers (32-32-64-64-128), GroupNorm, global average pooling, linear head (~0.14 M params)."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, widths=(32, 64, 128), groups: int = 8):
        super().__init__()
        w1, w2, w3 = widths
        self.body = nn.Sequential(
            *_cgr(in_ch, w1, groups), *_cgr(w1, w1, groups), nn.MaxPool2d(2),
            *_cgr(w1, w2, groups), *_cgr(w2, w2, groups), nn.MaxPool2d(2),
            *_cgr(w2, w3, groups), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(w3, num_classes)

    def forward(self, x):
        return self.head(self.body(x))


class MLP(nn.Module):
    """Two hidden layers (64, 32) + linear head, for tabular data."""

    def __init__(self, d_in: int, num_classes: int = 2, hidden=(64, 32)):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.head = nn.Linear(hidden[1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))


def build_model(cfg: dict, data) -> nn.Module:
    if cfg["model"] == "cnn":
        return CNN(num_classes=data.num_classes, in_ch=data.input_shape[0])
    if cfg["model"] == "mlp":
        return MLP(d_in=int(data.input_shape[0]), num_classes=data.num_classes)
    raise ValueError(f"unknown model {cfg['model']}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


@torch.no_grad()
def set_flat(model: nn.Module, vec: torch.Tensor) -> None:
    o = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(vec[o:o + n].view_as(p))
        o += n


def head_mask(model: nn.Module) -> torch.Tensor:
    """Boolean mask over the flat parameter vector: True where the parameter belongs to `head`."""
    parts = [torch.full((p.numel(),), name.startswith("head."), dtype=torch.bool)
             for name, p in model.named_parameters()]
    return torch.cat(parts)
