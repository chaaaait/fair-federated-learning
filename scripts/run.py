"""Run ONE experiment and save its JSON.

Examples (PowerShell, from the repo root):
  .venv\\Scripts\\python scripts\\run.py --dataset cifar10 --method fedavg --eps 2 --seed 0
  .venv\\Scripts\\python scripts\\run.py --method fedper --eps inf --set rounds=5 local_steps=5
"""
from __future__ import annotations

import argparse

from _common import BASE_YAML, RESULTS
from pfl.config import make_config, parse_set_args
from pfl.engine import run_experiment
from pfl.utils import eps_tag, write_json_atomic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--method", default="fedavg")
    ap.add_argument("--aggregation", default="mean")
    ap.add_argument("--eps", default="2", help="privacy budget; 'inf' = non-private")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides key=value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ov = {"dataset": args.dataset, "method": args.method, "aggregation": args.aggregation,
          "epsilon": args.eps, "seed": args.seed}
    ov.update(parse_set_args(args.set))
    cfg = make_config(ov, base_yaml=BASE_YAML)
    res = run_experiment(cfg)
    out = args.out or RESULTS / "adhoc" / f"{cfg['dataset']}_{cfg['method']}_{cfg['aggregation']}_eps{eps_tag(cfg['epsilon'])}_s{cfg['seed']}.json"
    write_json_atomic(out, res)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
