"""Why FedPer + q-FFL gives the same accuracy for q = 1 and q = 5 (level-2 review).

Replays the first rounds of the CIFAR-10 eps0 runs (same seed, and the sigma of the 40-round runs) and logs, per
round, the size of the server step on the shared body under q-FedAvg next to the plain mean of the same client
updates (what FedPer would apply).

  .venv\\Scripts\\python scripts\\diag_fedper_qffl.py      # -> results/diag_fedper_qffl.csv (~3 min on the GPU)
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from _common import BASE_YAML, RESULTS
from pfl import methods
from pfl.config import make_config
from pfl.data import build_federated_data
from pfl.dp import GradEngine, calibrate_sigma
from pfl.methods import Ctx, MeanAgg, query_schedule
from pfl.models import build_model, get_flat
from pfl.utils import resolve_device, set_seed


class Probe:
    """Wraps the q-FedAvg aggregator and records its step next to the unweighted mean of the same updates."""

    def __init__(self, inner, rows: list, q: float, w0: torch.Tensor):
        self.inner, self.mean, self.rows, self.q, self.w0 = inner, MeanAgg(1.0), rows, q, w0
        self.delta_norms, self.losses = [], []

    def add(self, delta, loss):
        self.inner.add(delta, loss)
        self.mean.add(delta, 1.0)
        self.delta_norms.append(float(delta.norm()))
        self.losses.append(float(loss))

    def result(self, w):
        new = self.inner.result(w)
        step, mean_step = float((new - w).norm()), float((self.mean.result(w) - w).norm())
        self.rows.append({"q": self.q, "round": len([r for r in self.rows if r["q"] == self.q]) + 1,
                          "client_update_norm_mean": sum(self.delta_norms) / len(self.delta_norms),
                          "loss_report_mean": sum(self.losses) / len(self.losses),
                          "mean_step_norm": mean_step, "qffl_step_norm": step, "ratio": step / mean_step,
                          "body_moved_from_init": float((new - self.w0).norm())})
        return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=2.0)
    args = ap.parse_args()
    rows = []
    for q in (1, 5):
        cfg = make_config({"dataset": "cifar10", "method": "fedper", "aggregation": "qffl", "qffl_q": q,
                           "epsilon": args.eps, "seed": args.seed, "mia": False}, base_yaml=BASE_YAML)
        device = resolve_device(cfg["device"])
        set_seed(cfg["seed"])
        data = build_federated_data(cfg, device)
        set_seed(cfg["seed"])
        model = build_model(cfg, data).to(device)
        w0 = get_flat(model)
        t_grad, t_loss = query_schedule(cfg)  # the full 40-round schedule -> the sigma of the real runs
        sigma = calibrate_sigma(cfg["epsilon"], cfg["sample_rate"], t_grad, t_loss, cfg["loss_noise_ratio"],
                                cfg["delta"])
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg["seed"]) * 7919 + 17)
        ctx = Ctx(cfg, data, model, GradEngine(model, cfg["grad_backend"], cfg["max_physical_batch"]), device,
                  sigma, gen, log=print)
        ctx.R = args.rounds
        inner_factory = methods._aggregator
        # the server vector's head entries never change (FedPer masks them out), so |w - w0| is the body's move
        methods._aggregator = lambda c, q=q: Probe(inner_factory(c), rows, q, w0)
        try:
            print(f"q = {q}: sigma = {sigma:.3f}, replaying {args.rounds} rounds")
            methods.run_fedper(ctx, w0)
        finally:
            methods._aggregator = inner_factory
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "diag_fedper_qffl.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
