"""Default configuration. Every value can be overridden from YAML or the command line.

Starting values come from PLAN §3 (design defaults D1-D10). Tune only on the validation
split; record final values in paper Table T3.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .utils import parse_eps

DEFAULTS = {
    # data
    "dataset": "cifar10",            # cifar10 | adult | synthetic | synthetic_tab
    "data_dir": "data",
    "num_clients": 20,               # image datasets (Adult uses one client per occupation)
    "dirichlet_alpha": 0.3,          # label skew; smaller = more heterogeneous
    "min_client_size": 300,          # re-draw the Dirichlet partition until every client has this many
    "adult_min_rows": 500,           # occupations with fewer rows are dropped
    "split": [0.7, 0.1, 0.2],        # per-client train / validation / test
    "partition_seed": 42,            # fixed across training seeds -> same clients for every method
    "max_examples": None,            # image datasets: random subset of this size (live demo only)
    # model
    "model": "auto",                 # auto | cnn | mlp
    # federated training
    "method": "fedavg",              # fedavg | local | perfedavg | ifca | fedper | mrmtl
    "aggregation": "mean",           # mean | qffl
    "qffl_q": 0.0,
    "qffl_L_scale": 1.0,             # q-FedAvg Lipschitz estimate L = scale / lr_local (1.0 = Li et al. default)
    "ft_branch": True,               # fedavg only: also produce the fine-tuning configuration (D10)
    "ft_fraction": 0.1,              # share of each client's step budget spent on local fine-tuning
    "rounds": 40,
    "local_steps": 20,               # private gradient queries per client per round
    "lr_local": 0.05,
    "lr_server": 1.0,
    "perfedavg_alpha": 0.03,         # inner (adaptation) step size
    "perfedavg_beta": 0.05,          # outer (meta) step size
    "perfedavg_adapt_steps": 5,      # personalisation steps after training (charged to the budget)
    "perfedavg_meta_steps": None,    # meta-steps per round (2 queries each); None = local_steps // 2
    "ifca_k": 3,                     # number of cluster models
    "mrmtl_lambda": 0.1,             # MR-MTL pull towards the server mean; the queue picks it on validation
    # differential privacy (per-example DP-SGD inside each client)
    "epsilon": 2.0,                  # null / inf = non-private
    "delta": 1.0e-5,
    "sample_rate": 0.03,             # Poisson sampling rate per record per step
    "clip_norm": 1.0,
    "loss_noise_ratio": 30.0,        # noise multiplier of private loss reports = ratio x sigma
    "loss_clip": 3.0,                # per-record loss clip for private loss reports
    "clip_only": False,              # control (eps = inf): clip and Poisson-sample as in DP-SGD, add no noise
    "grad_backend": "opacus",        # opacus | functorch
    "max_physical_batch": 128,       # per-example gradients are computed in chunks of this size
    # evaluation
    "eval_every": 10,
    "mia": True,
    # run
    "seed": 0,
    "device": "auto",
}

DATASET_OVERRIDES = {
    "adult": {"model": "mlp", "rounds": 30, "local_steps": 10, "lr_local": 0.1},
    "synthetic_tab": {"model": "mlp", "rounds": 5, "local_steps": 4, "lr_local": 0.1},
    "synthetic": {"rounds": 3, "local_steps": 3, "num_clients": 4, "min_client_size": 40,
                  "sample_rate": 0.2, "eval_every": 1},
}


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_config(overrides: dict | None = None, base_yaml: str | Path | None = None) -> dict:
    """DEFAULTS <- base.yaml <- dataset overrides <- explicit overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if base_yaml is not None and Path(base_yaml).exists():
        cfg.update(load_yaml(base_yaml).get("defaults", {}))
    overrides = dict(overrides or {})
    dataset = overrides.get("dataset", cfg["dataset"])
    cfg.update(DATASET_OVERRIDES.get(dataset, {}))
    if base_yaml is not None and Path(base_yaml).exists():
        cfg.update((load_yaml(base_yaml).get("dataset_overrides", {}) or {}).get(dataset, {}))
    cfg.update(overrides)
    cfg["epsilon"] = parse_eps(cfg.get("epsilon"))
    if cfg["model"] == "auto":
        cfg["model"] = "mlp" if cfg["dataset"] in ("adult", "synthetic_tab") else "cnn"
    return cfg


def parse_set_args(pairs: list[str]) -> dict:
    """Parse ['key=value', ...] with YAML typing (numbers, booleans, null)."""
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out
