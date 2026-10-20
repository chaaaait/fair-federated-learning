"""Small helpers: seeding, device, timing, JSON I/O, version capture."""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Stopwatch:
    """Wall-clock timer that waits for queued GPU work before reading the clock."""

    def __init__(self, device: torch.device):
        self.device = device
        self.t0 = None

    def start(self) -> None:
        sync(self.device)
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        sync(self.device)
        return time.perf_counter() - self.t0


def is_private(eps) -> bool:
    return eps is not None and not (isinstance(eps, float) and math.isinf(eps))


def parse_eps(value):
    """Return None for a non-private run ('inf', None, math.inf) or a float budget."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("inf", "infinity", "none", "null", "nonprivate"):
            return None
        value = float(value)
    value = float(value)
    return None if math.isinf(value) else value


def eps_tag(eps) -> str:
    eps = parse_eps(eps)
    return "inf" if eps is None else f"{eps:g}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def versions() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "numpy": np.__version__,
    }
    try:
        import opacus

        info["opacus"] = opacus.__version__
    except Exception:  # pragma: no cover
        info["opacus"] = None
    try:
        import flwr

        info["flwr"] = flwr.__version__
    except Exception:
        info["flwr"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
