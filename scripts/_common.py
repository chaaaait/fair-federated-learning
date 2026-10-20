"""Shared helpers for the command-line scripts: repo paths and queue expansion."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfl.config import load_yaml, make_config  # noqa: E402
from pfl.utils import eps_tag, parse_eps  # noqa: E402

RESULTS = Path(os.environ.get("PFL_RESULTS", ROOT / "results"))  # override for tests
RUNS = RESULTS / "runs"
LOGS = Path(os.environ.get("PFL_LOGS", ROOT / "logs"))
BASE_YAML = ROOT / "configs" / "base.yaml"
QUEUE_YAML = Path(os.environ.get("PFL_QUEUE", ROOT / "configs" / "queue.yaml"))


@dataclass
class RunSpec:
    run_id: str
    block: str
    overrides: dict = field(default_factory=dict)

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
    return label


def expand_queue(queue_path: Path = QUEUE_YAML, only: list[str] | None = None) -> list[RunSpec]:
    """Queue YAML -> ordered list of unique runs (blocks top to bottom, then eps, run, seed)."""
    q = load_yaml(queue_path)
    eps0 = q.get("eps0", 2.0)
    seeds = q.get("seeds", [0, 1, 2])
    specs, seen = [], {}
    for block in q["blocks"]:
        if only and block["name"] not in only:
            continue
        for eps_raw in block["eps"]:
            eps = parse_eps(eps0 if str(eps_raw) == "eps0" else eps_raw)
            for run in block["runs"]:
                for seed in block.get("seeds", seeds):
                    run_id = f"{block['dataset']}__{label_of(run)}__eps{eps_tag(eps)}__s{seed}"
                    if run_id in seen:
                        seen[run_id].overrides.setdefault("_blocks", []).append(block["name"])
                        continue
                    ov = dict(run)
                    ov.update({"dataset": block["dataset"], "epsilon": eps, "seed": int(seed),
                               "_blocks": [block["name"]]})
                    spec = RunSpec(run_id=run_id, block=block["name"], overrides=ov)
                    seen[run_id] = spec
                    specs.append(spec)
    return specs


def config_for(spec: RunSpec) -> dict:
    ov = {k: v for k, v in spec.overrides.items() if not k.startswith("_")}
    cfg = make_config(ov, base_yaml=BASE_YAML)
    cfg["run_id"] = spec.run_id
    cfg["blocks"] = spec.overrides.get("_blocks", [spec.block])
    return cfg
