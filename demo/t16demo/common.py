"""Shared helpers for the demo's ServerApp and ClientApp (both reuse the `pfl` library)."""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch

from pfl.config import make_config
from pfl.models import build_model


def demo_config(run_config) -> dict:
    """Translate the Flower run config (pyproject.toml [tool.flwr.app.config]) into a pfl config."""
    ov = {
        "dataset": str(run_config["dataset"]),
        "num_clients": int(run_config["num-partitions"]),
        "rounds": int(run_config["num-server-rounds"]),
        "local_steps": int(run_config["local-steps"]),
        "epsilon": float(run_config["epsilon"]),
        "delta": float(run_config["delta"]),
        "sample_rate": float(run_config["sample-rate"]),
        "clip_norm": float(run_config["clip-norm"]),
        "lr_local": float(run_config["lr"]),
        "device": str(run_config["device"]),
        "seed": int(run_config["seed"]),
        "max_examples": int(run_config["max-examples"]) or None,
        "min_client_size": 100,
        "method": "fedper",
    }
    if str(run_config["data-dir"]):
        ov["data_dir"] = str(run_config["data-dir"])
    if ov["dataset"] == "synthetic":  # CPU self-test
        ov.update({"max_examples": None, "min_client_size": 40, "sample_rate": 0.2})
    return make_config(ov)


def fresh_model(cfg) -> torch.nn.Module:
    """Model with the experiment's architecture; same seed -> same initial body on server and clients."""
    torch.manual_seed(cfg["seed"])
    shape = (3, 16, 16) if cfg["dataset"] == "synthetic" else (3, 32, 32)
    stub = SimpleNamespace(num_classes=10, input_shape=shape)
    return build_model(cfg, stub)


def split_state(model) -> tuple[OrderedDict, OrderedDict]:
    """(body, head) state dicts. FedPer shares only the body."""
    body, head = OrderedDict(), OrderedDict()
    for name, tensor in model.state_dict().items():
        (head if name.startswith("head.") else body)[name] = tensor.detach().cpu()
    return body, head
