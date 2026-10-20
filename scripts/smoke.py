"""Gate 1 checks on the real machine (PLAN §4). Takes ~5-10 min on the GTX 1650.

  .venv\\Scripts\\python scripts\\smoke.py              # CIFAR-10 (downloads ~170 MB on first run)
  .venv\\Scripts\\python scripts\\smoke.py --dataset adult

Checks
  1. GPU visible and supported (sm_75 for a GTX 1650).
  2. Opacus and torch.func per-example gradients agree.
  3. Every method runs 2 private rounds without error.
  4. Accountant: spent epsilon within 0.05 of the target.
  5. Timing: per-round time x rounds -> estimated minutes per DP run (target <= 6).
Writes results/smoke.json and prints PASS / FAIL lines.
"""
from __future__ import annotations

import argparse
import time

import torch

from _common import BASE_YAML, RESULTS
from pfl.config import make_config
from pfl.data import build_federated_data
from pfl.dp import GradEngine
from pfl.engine import run_experiment
from pfl.models import build_model
from pfl.utils import resolve_device, versions, write_json_atomic

CASES = [
    {"method": "fedavg"},
    {"method": "fedavg", "aggregation": "qffl", "qffl_q": 1},
    {"method": "local"},
    {"method": "perfedavg"},
    {"method": "ifca"},
    {"method": "fedper"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--eps", default="2")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    report = {"versions": versions(), "checks": {}}
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    dev = resolve_device("auto")
    arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    check("gpu", dev.type == "cuda", f"device={dev} gpu={gpu} arch_list={arch}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        check("gpu_arch_supported", f"sm_{major}{minor}" in arch or any(a.startswith(f"sm_{major}") for a in arch),
              f"capability sm_{major}{minor}")

    base = make_config({"dataset": args.dataset}, base_yaml=BASE_YAML)
    full_rounds = base["rounds"]
    t = time.perf_counter()
    data = build_federated_data(base, dev)
    sizes = [c.n_tr for c in data.clients]
    check("data", True, f"{len(sizes)} clients, train sizes min {min(sizes)} / median {sorted(sizes)[len(sizes)//2]} "
                        f"/ max {max(sizes)} ({time.perf_counter() - t:.0f}s)")

    torch.manual_seed(0)
    m1 = build_model(base, data).to(dev)
    torch.manual_seed(0)
    m2 = build_model(base, data).to(dev)
    c = data.clients[0]
    xb, yb = data.prep(c.x_tr[:16]), c.y_tr[:16]
    try:
        g1 = GradEngine(m1, "opacus").per_sample_grads(xb, yb)
        g2 = GradEngine(m2, "functorch").per_sample_grads(xb, yb)
        diff = (g1 - g2).abs().max().item()
        check("backends_agree", diff < 1e-3, f"max |opacus - functorch| = {diff:.2e}")
    except Exception as exc:
        check("backends_agree", False, f"{exc!r} -> set grad_backend: functorch in configs/base.yaml")

    timings = {}
    for case in CASES:
        cfg = make_config({"dataset": args.dataset, "epsilon": args.eps, "rounds": args.rounds,
                           "eval_every": args.rounds, **case}, base_yaml=BASE_YAML)
        name = case["method"] + ("-qffl" if case.get("aggregation") == "qffl" else "")
        try:
            res = run_experiment(cfg, log=lambda *a, **k: None)
            per_round = res["cost"]["time_per_round_s"]
            timings[name] = per_round
            est_min = per_round * full_rounds / 60 * 1.1  # +10% for evaluation
            check(f"run_{name}", True, f"{per_round:.1f}s/round -> ~{est_min:.1f} min for {full_rounds} rounds, "
                                       f"eps {res['privacy']['epsilon_spent']:.3f}, peak GPU {res['cost']['peak_gpu_MB']} MB")
            target = float(args.eps)
            check(f"accountant_{name}", abs(res["privacy"]["epsilon_spent"] - target) <= 0.05,
                  f"spent {res['privacy']['epsilon_spent']:.4f} vs target {target}")
        except Exception as exc:
            check(f"run_{name}", False, repr(exc))

    if timings:
        worst = max(timings.values()) * full_rounds / 60 * 1.1
        check("time_budget", worst <= 6.0, f"slowest method ~{worst:.1f} min per run (target <= 6; "
                                           "if over, lower rounds in configs/base.yaml)")
    report["timings_s_per_round"] = timings
    report["all_ok"] = ok_all
    write_json_atomic(RESULTS / f"smoke_{args.dataset}.json", report)
    print("\nGATE 1:", "PASS" if ok_all else "FAIL", f"(details in results/smoke_{args.dataset}.json)")


if __name__ == "__main__":
    main()
