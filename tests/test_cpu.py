"""CPU tests on generated data (no downloads). Run: python tests/test_cpu.py  (or pytest tests/)

Checks: every method runs privately and non-privately, the accountant hits the target epsilon,
both per-example gradient backends agree, every client answers the same number of queries, and
group-fairness metrics appear for tabular data.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pfl.config import make_config  # noqa: E402
from pfl.data import build_federated_data  # noqa: E402
from pfl.dp import GradEngine, calibrate_sigma, epsilon_from  # noqa: E402
from pfl.engine import run_experiment  # noqa: E402
from pfl.methods import query_schedule  # noqa: E402
from pfl.models import build_model  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731

CASES = [
    {"method": "fedavg"},
    {"method": "fedavg", "aggregation": "qffl", "qffl_q": 1},
    {"method": "local"},
    {"method": "perfedavg"},
    {"method": "ifca"},
    {"method": "fedper"},
    {"method": "fedper", "aggregation": "qffl", "qffl_q": 5},
]


def _cfg(**ov):
    base = {"dataset": "synthetic", "device": "cpu", "data_dir": str(ROOT / "data")}
    base.update(ov)
    return make_config(base)


def test_accountant():
    sigma = calibrate_sigma(2.0, 0.03, 800, 0)
    eps = epsilon_from(sigma, 0.03, 800, 0)
    assert 1.95 <= eps <= 2.0, eps
    sigma_l = calibrate_sigma(2.0, 0.03, 800, 120)
    assert sigma_l > sigma  # loss reports cost budget -> more gradient noise


def test_backends_agree():
    cfg = _cfg()
    data = build_federated_data(cfg, torch.device("cpu"))
    torch.manual_seed(0)
    model = build_model(cfg, data)
    c = data.clients[0]
    xb, yb = data.prep(c.x_tr[:8]), c.y_tr[:8]
    g_op = GradEngine(model, "opacus").per_sample_grads(xb, yb)
    torch.manual_seed(0)
    model2 = build_model(cfg, data)
    g_ft = GradEngine(model2, "functorch").per_sample_grads(xb, yb)
    assert g_op.shape == g_ft.shape
    assert torch.allclose(g_op, g_ft, atol=1e-5, rtol=1e-4), (g_op - g_ft).abs().max()


def test_methods_private_and_nonprivate():
    for case in CASES:
        for eps in (4.0, None):
            cfg = _cfg(epsilon=eps, **case)
            t0 = time.perf_counter()
            res = run_experiment(cfg, log=QUIET)
            t_grad, t_loss = query_schedule(cfg)
            obs = res["privacy"]["observed_queries_per_client"]
            assert len(obs) == 1, f"clients answered different query counts: {obs}"
            if eps is not None:
                assert abs(res["privacy"]["epsilon_spent"] - eps) < 0.05, res["privacy"]
            for name, v in res["views"].items():
                assert len(v["acc_test"]) == len(res["data"]["clients"]), name
                assert 0.0 <= v["summary"]["mean"] <= 1.0
                assert "mia" in v
            print(f"  ok {case} eps={eps} views={list(res['views'])} "
                  f"({time.perf_counter() - t0:.1f}s) mean={[round(v['summary']['mean'], 3) for v in res['views'].values()]}")


def test_tabular_group_metrics():
    cfg = _cfg(dataset="synthetic_tab", method="fedavg", epsilon=4.0)
    res = run_experiment(cfg, log=QUIET)
    g = res["views"]["fedavg"]["group"]
    assert not math.isnan(g["pooled"]["dp_gap"])
    print(f"  ok tabular: {len(res['data']['clients'])} clients, pooled gaps {g['pooled']}")


if __name__ == "__main__":
    for fn in (test_accountant, test_backends_agree, test_methods_private_and_nonprivate, test_tabular_group_metrics):
        t = time.perf_counter()
        fn()
        print(f"PASS {fn.__name__} ({time.perf_counter() - t:.1f}s)")
