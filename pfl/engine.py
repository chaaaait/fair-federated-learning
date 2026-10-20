"""Run one experiment end to end and return a JSON-ready result dictionary."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .data import build_federated_data
from .dp import GradEngine, calibrate_sigma, epsilon_from
from .methods import METHODS, Ctx, query_schedule
from .metrics import fairness_summary, group_gaps, logits_of, loss_threshold_mia
from .models import build_model, get_flat, num_params, set_flat
from .utils import eps_tag, resolve_device, set_seed, versions


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def evaluate_view(ctx: Ctx, view: dict, rng: np.random.Generator, with_mia: bool) -> dict:
    accs, mias, preds = [], [], []
    loaded = None
    for i, c in enumerate(ctx.clients):
        vec = view["vecs"][i]
        if vec is not loaded:
            set_flat(ctx.model, vec)
            loaded = vec
        yhat = logits_of(ctx.model, c.x_te, ctx.data.prep).argmax(1)
        accs.append(float((yhat == c.y_te).float().mean().item()))
        if with_mia:
            mias.append(loss_threshold_mia(ctx.model, c, ctx.data.prep, rng))
        if c.s_te is not None:
            preds.append((yhat.cpu().numpy(), c.y_te.cpu().numpy(), c.s_te.cpu().numpy()))
    out = {"acc_test": accs, "summary": fairness_summary(accs), "comm": view["comm"]}
    if mias:
        out["mia"] = {
            "auc_mean": _nanmean([m["auc"] for m in mias]),
            "acc_mean": _nanmean([m["acc"] for m in mias]),
            "adv_mean": _nanmean([m["adv"] for m in mias]),
            "auc": [m["auc"] for m in mias],
        }
    if preds:
        per = [group_gaps(*p) for p in preds]
        pooled = group_gaps(np.concatenate([p[0] for p in preds]), np.concatenate([p[1] for p in preds]),
                            np.concatenate([p[2] for p in preds]))
        out["group"] = {"pooled": pooled, "per_client": per,
                        "dp_gap_mean": _nanmean([g["dp_gap"] for g in per]),
                        "eo_gap_mean": _nanmean([g["eo_gap"] for g in per])}
    for key in ("assign_hist", "final_assign", "note", "global"):
        if key in view:
            out[key] = view[key]
    return out


def run_experiment(cfg: dict, log=print) -> dict:
    t0 = time.perf_counter()
    device = resolve_device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(cfg["seed"])
    data = build_federated_data(cfg, device)
    set_seed(cfg["seed"])
    model = build_model(cfg, data).to(device)
    w0 = get_flat(model)

    eps = cfg["epsilon"]
    t_grad, t_loss = query_schedule(cfg)
    if eps is None:
        sigma, eps_spent = 0.0, None
    else:
        sigma = calibrate_sigma(eps, cfg["sample_rate"], t_grad, t_loss, cfg["loss_noise_ratio"], cfg["delta"])
        eps_spent = epsilon_from(sigma, cfg["sample_rate"], t_grad, t_loss, cfg["loss_noise_ratio"], cfg["delta"])

    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg["seed"]) * 7919 + 17)
    engine = GradEngine(model, cfg["grad_backend"], cfg["max_physical_batch"])
    ctx = Ctx(cfg, data, model, engine, device, sigma, gen, log=log)
    log(f"[run] {cfg['dataset']} {cfg['method']}/{cfg['aggregation']} eps={eps_tag(eps)} seed={cfg['seed']} "
        f"sigma={sigma:.3f} queries/client={t_grad}+{t_loss} K={ctx.K} P={ctx.P} device={device}")

    views = METHODS[cfg["method"]](ctx, w0)

    rng = np.random.default_rng(int(cfg["seed"]) + 2026)
    out_views = {name: evaluate_view(ctx, v, rng, cfg["mia"]) for name, v in views.items()}
    observed = sorted({(v["grad"], v["loss"]) for v in ctx.qcount.values()})

    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    total = time.perf_counter() - t0
    result = {
        "status": "ok",
        "config": cfg,
        "privacy": {
            "private": eps is not None,
            "epsilon_target": eps,
            "epsilon_spent": eps_spent,
            "delta": cfg["delta"],
            "sigma": sigma,
            "sigma_loss": ctx.sigma_loss,
            "sample_rate": cfg["sample_rate"],
            "clip_norm": cfg["clip_norm"],
            "grad_queries_per_client": t_grad,
            "loss_queries_per_client": t_loss,
            "observed_queries_per_client": observed,
        },
        "data": data.meta,
        "model": {"name": cfg["model"], "params": ctx.P, "head_params": ctx.P_head},
        "views": out_views,
        "curves": ctx.curves,
        "cost": {
            "time_total_s": total,
            "time_train_s": float(np.sum(ctx.round_times)),
            "time_per_round_s": float(np.mean(ctx.round_times)) if ctx.round_times else None,
            "rounds": ctx.R,
            "peak_gpu_MB": peak,
            "device": str(device),
        },
        "versions": versions(),
    }
    for name, v in out_views.items():
        s = v["summary"]
        mia = v.get("mia", {}).get("auc_mean", float("nan"))
        log(f"[done] {name:<16} mean {s['mean']:.3f} worst10 {s['worst10']:.3f} jain {s['jain']:.3f} "
            f"MIA-AUC {mia:.3f} | {total:.0f}s")
    return result
