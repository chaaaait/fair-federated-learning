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
    {"method": "mrmtl", "mrmtl_lambda": 1.0},
    {"method": "perfedavg", "perfedavg_meta_steps": 3},
    {"method": "ifca", "ifca_k": 2},
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


def test_mrmtl_lambda0_is_local():
    """lambda = 0 removes the pull towards the mean, so MR-MTL must reproduce local training exactly."""
    for eps in (4.0, None):
        a = run_experiment(_cfg(method="mrmtl", mrmtl_lambda=0.0, epsilon=eps), log=QUIET)
        b = run_experiment(_cfg(method="local", epsilon=eps), log=QUIET)
        assert a["views"]["mrmtl"]["acc_test"] == b["views"]["local"]["acc_test"], eps
        assert a["privacy"]["sigma"] == b["privacy"]["sigma"]
    print("  ok MR-MTL with lambda = 0 equals local training (private and non-private)")


def test_clip_only():
    """clip_only clips like DP-SGD without noise: eps = inf, sigma = 0, and a different result than plain SGD."""
    c = run_experiment(_cfg(method="ifca", clip_only=True, epsilon=None), log=QUIET)
    p = run_experiment(_cfg(method="ifca", epsilon=None), log=QUIET)
    assert c["privacy"]["clip_only"] and not p["privacy"]["clip_only"]
    assert c["privacy"]["sigma"] == 0.0 and c["privacy"]["epsilon_spent"] is None
    assert c["views"]["ifca"]["acc_test"] != p["views"]["ifca"]["acc_test"]
    try:
        run_experiment(_cfg(method="fedavg", clip_only=True, epsilon=4.0), log=QUIET)
        raise AssertionError("clip_only with a finite epsilon should fail")
    except ValueError:
        pass
    print("  ok clip-only control")


def test_query_schedule_options():
    cfg = _cfg(method="perfedavg", perfedavg_meta_steps=5)
    assert query_schedule(cfg) == (2 * 5 * cfg["rounds"] + cfg["perfedavg_adapt_steps"], 0)
    cfg = _cfg(method="perfedavg")
    assert query_schedule(cfg) == (2 * (cfg["local_steps"] // 2) * cfg["rounds"] + cfg["perfedavg_adapt_steps"], 0)
    cfg = _cfg(method="ifca", ifca_k=2)
    assert query_schedule(cfg) == (cfg["rounds"] * cfg["local_steps"], 2 * cfg["rounds"])
    cfg = _cfg(method="mrmtl")
    assert query_schedule(cfg) == (cfg["rounds"] * cfg["local_steps"], 0)
    print("  ok query schedules")


def test_run_ids_and_auto_lambda():
    """Old run IDs must not change; new options get their own suffix; 'auto' picks lambda on validation only."""
    import json
    import os
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ["PFL_RESULTS"] = str(tmp)
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    import _common
    importlib.reload(_common)
    lo = _common.label_of
    assert lo({"method": "fedavg"}) == "fedavg"
    assert lo({"method": "fedavg", "aggregation": "qffl", "qffl_q": 1, "qffl_L_scale": 0.01}) == "fedavg-qffl1-L0.01"
    assert lo({"method": "fedper", "dirichlet_alpha": 1.0}) == "fedper-a1"
    assert lo({"method": "fedavg", "rounds": 200}) == "fedavg-R200"
    assert lo({"method": "fedavg", "clip_only": True}) == "fedavg-clip"
    assert lo({"method": "local", "lr_local": 0.025}) == "local-lr0.025"
    assert lo({"method": "perfedavg", "perfedavg_meta_steps": 20}) == "perfedavg-ms20"
    assert lo({"method": "ifca", "ifca_k": 2}) == "ifca-k2"
    assert lo({"method": "mrmtl", "mrmtl_lambda": 0.1}) == "mrmtl-lam0.1"
    assert lo({"method": "mrmtl", "mrmtl_lambda": "auto"}) == "mrmtl-lamauto"

    queue = tmp / "q.yaml"
    queue.write_text("eps0: 2.0\nseeds: [0]\nmrmtl_grid: [0.1, 1]\nmrmtl_select_seeds: [0]\nblocks:\n"
                     "  - {name: G, dataset: synthetic, eps: [eps0], runs: [{method: mrmtl, mrmtl_lambda: 0.1}, "
                     "{method: mrmtl, mrmtl_lambda: 1}]}\n"
                     "  - {name: A, dataset: synthetic, eps: [eps0], seeds: [1], runs: [{method: mrmtl, mrmtl_lambda: auto}]}\n",
                     encoding="utf-8")
    specs = _common.expand_queue(queue)
    assert [s.waiting for s in specs] == [False, False, True], [(s.run_id, s.waiting) for s in specs]
    (tmp / "runs").mkdir(parents=True, exist_ok=True)
    for lam, val in ((0.1, 0.40), (1.0, 0.55)):  # validation says lambda = 1 (test accuracy is never read)
        p = tmp / "runs" / f"synthetic__mrmtl-lam{lam:g}__eps2__s0.json"
        p.write_text(json.dumps({"curves": {"val_mean": [0.1, val]}}), encoding="utf-8")
    specs = _common.expand_queue(queue)
    assert specs[-1].run_id == "synthetic__mrmtl-lam1__eps2__s1" and not specs[-1].waiting, specs[-1]
    assert specs[-1].overrides["mrmtl_lambda"] == 1.0
    del os.environ["PFL_RESULTS"]
    print("  ok run IDs and auto lambda")


if __name__ == "__main__":
    for fn in (test_accountant, test_backends_agree, test_methods_private_and_nonprivate, test_tabular_group_metrics,
               test_mrmtl_lambda0_is_local, test_clip_only, test_query_schedule_options, test_run_ids_and_auto_lambda):
        t = time.perf_counter()
        fn()
        print(f"PASS {fn.__name__} ({time.perf_counter() - t:.1f}s)")
