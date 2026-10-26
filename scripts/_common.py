"""Shared helpers for the command-line scripts: repo paths and queue expansion."""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfl.config import load_yaml, make_config  # noqa: E402
from pfl.utils import eps_tag, parse_eps, read_json  # noqa: E402

RESULTS = Path(os.environ.get("PFL_RESULTS", ROOT / "results"))  # override for tests
RUNS = RESULTS / "runs"
LOGS = Path(os.environ.get("PFL_LOGS", ROOT / "logs"))
BASE_YAML = ROOT / "configs" / "base.yaml"
QUEUE_YAML = Path(os.environ.get("PFL_QUEUE", ROOT / "configs" / "queue.yaml"))


AUTO = "auto"  # mrmtl_lambda: auto = the grid value with the best validation accuracy (see resolve_auto_lambda)


@dataclass
class RunSpec:
    run_id: str
    block: str
    overrides: dict = field(default_factory=dict)
    waiting: bool = False  # an "auto" run whose selection runs have not all finished yet

    @property
    def result_path(self) -> Path:
        return RUNS / f"{self.run_id}.json"

    @property
    def error_path(self) -> Path:
        return RUNS / f"{self.run_id}.error.json"


def label_of(run: dict) -> str:
    label = run["method"]
    if run.get("aggregation", "mean") == "qffl":
        label += f"-qffl{float(run.get('qffl_q', 0)):g}"
    if "dirichlet_alpha" in run:
        label += f"-a{float(run['dirichlet_alpha']):g}"
    if float(run.get("qffl_L_scale", 1.0)) != 1.0:
        label += f"-L{float(run['qffl_L_scale']):g}"
    if "rounds" in run:
        label += f"-R{int(run['rounds'])}"
    if run.get("clip_only"):
        label += "-clip"
    if "clip_norm" in run:
        label += f"-C{float(run['clip_norm']):g}"
    if "lr_local" in run:
        label += f"-lr{float(run['lr_local']):g}"
    if "perfedavg_meta_steps" in run:
        label += f"-ms{int(run['perfedavg_meta_steps'])}"
    if "ifca_k" in run:
        label += f"-k{int(run['ifca_k'])}"
    if "mrmtl_lambda" in run:
        lam = run["mrmtl_lambda"]
        label += f"-lam{lam}" if str(lam) == AUTO else f"-lam{float(lam):g}"
    return label


def run_path(dataset: str, run: dict, eps, seed) -> Path:
    return RUNS / f"{dataset}__{label_of(run)}__eps{eps_tag(eps)}__s{seed}.json"


def final_val_mean(path: Path) -> float | None:
    """Mean validation accuracy over clients at the last logged round; None if the run has not finished."""
    if not path.exists():
        return None
    vals = read_json(path)["curves"]["val_mean"]
    return float(vals[-1]) if vals else None


def resolve_auto_lambda(run: dict, dataset: str, eps, grid: list, select_seeds: list) -> float | None:
    """MR-MTL lambda chosen on validation: the grid value whose runs (same dataset, eps and other settings) have the
    highest final validation accuracy, averaged over `select_seeds`; ties go to the smaller value. None until every
    grid run has finished. Test accuracy is never looked at."""
    best, best_val = None, -math.inf
    for lam in sorted(float(x) for x in grid):
        vals = [final_val_mean(run_path(dataset, dict(run, mrmtl_lambda=lam), eps, s)) for s in select_seeds]
        if not vals or any(v is None for v in vals):
            return None
        val = sum(vals) / len(vals)
        if val > best_val:
            best, best_val = lam, val
    return best


def expand_queue(queue_path: Path = QUEUE_YAML, only: list[str] | None = None) -> list[RunSpec]:
    """Queue YAML -> ordered list of unique runs (blocks top to bottom, then eps, run, seed)."""
    q = load_yaml(queue_path)
    eps0 = q.get("eps0", 2.0)
    seeds = q.get("seeds", [0, 1, 2])
    grid, select_seeds = q.get("mrmtl_grid", []), q.get("mrmtl_select_seeds", [0])
    specs, seen = [], {}
    for block in q["blocks"]:
        if only and block["name"] not in only:
            continue
        for eps_raw in block["eps"]:
            eps = parse_eps(eps0 if str(eps_raw) == "eps0" else eps_raw)
            for run in block["runs"]:
                waiting = False
                if str(run.get("mrmtl_lambda")) == AUTO:
                    lam = resolve_auto_lambda(run, block["dataset"], eps, grid, select_seeds)
                    if lam is None:
                        waiting = True
                    else:
                        run = dict(run, mrmtl_lambda=lam)
                for seed in block.get("seeds", seeds):
                    run_id = f"{block['dataset']}__{label_of(run)}__eps{eps_tag(eps)}__s{seed}"
                    if run_id in seen:
                        seen[run_id].overrides.setdefault("_blocks", []).append(block["name"])
                        continue
                    ov = dict(run)
                    ov.update({"dataset": block["dataset"], "epsilon": eps, "seed": int(seed),
                               "_blocks": [block["name"]]})
                    spec = RunSpec(run_id=run_id, block=block["name"], overrides=ov, waiting=waiting)
                    seen[run_id] = spec
                    specs.append(spec)
    return specs


def config_for(spec: RunSpec) -> dict:
    ov = {k: v for k, v in spec.overrides.items() if not k.startswith("_")}
    cfg = make_config(ov, base_yaml=BASE_YAML)
    cfg["run_id"] = spec.run_id
    cfg["blocks"] = spec.overrides.get("_blocks", [spec.block])
    return cfg
