"""Collect results/runs/*.json into tables (CSV) used by figures.py and the paper.

  .venv\\Scripts\\python scripts\\aggregate.py

Outputs (results/):
  summary_runs.csv     one row per (run, view) = configuration x epsilon x seed
  summary_table.csv    mean and SD over seeds for every metric
  eps_slopes.csv       accuracy loss per unit epsilon between neighbouring budgets, and drop vs non-private
  recommended.csv      problem statement Eq. (5): per epsilon, the mechanism with the largest mean
                       personalisation gain whose Jain index is >= DP-FedAvg's (the fairness floor J0)
  sanity.txt           Gate 3 checks
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from _common import BASE_YAML, RESULTS, RUNS
from pfl.config import make_config

PERSONALISED = ["local", "finetune", "perfedavg", "ifca", "fedper"]
METRICS = ["mean", "worst10", "std", "var", "jain", "min", "gain_mean", "gain_worst10", "frac_hurt",
           "mia_auc", "mia_acc", "mia_adv", "comm_total_MB", "comm_MB_per_round", "time_per_round_s",
           "time_total_s", "peak_gpu_MB", "dp_gap", "eo_gap", "dp_gap_mean", "eo_gap_mean",
           "eps_spent", "sigma"]


def eps_value(e):
    return math.inf if e is None else float(e)


_DEFAULT_ROUNDS = {}


def default_rounds(dataset: str) -> int:
    """Rounds of the main protocol (configs/base.yaml); runs with other round counts are kept apart."""
    if dataset not in _DEFAULT_ROUNDS:
        _DEFAULT_ROUNDS[dataset] = int(make_config({"dataset": dataset}, base_yaml=BASE_YAML)["rounds"])
    return _DEFAULT_ROUNDS[dataset]


def load_rows() -> tuple[pd.DataFrame, dict]:
    """Rows for every (run, view). per_client holds main-protocol runs only; runs with a non-default round
    count (convergence checks) are in df.attrs["per_client_conv"], keyed the same way plus the round count."""
    rows, per_client, per_client_conv = [], {}, {}
    for path in sorted(RUNS.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        cfg, priv, cost = r["config"], r["privacy"], r["cost"]
        main_protocol = int(cfg["rounds"]) == default_rounds(cfg["dataset"])
        for view, v in r["views"].items():
            s = v["summary"]
            mia = v.get("mia", {})
            grp = v.get("group", {})
            key = (cfg["dataset"], view, eps_value(priv["epsilon_target"]), cfg["seed"], cfg["dirichlet_alpha"])
            if main_protocol:
                per_client[key] = np.asarray(v["acc_test"], dtype=float)
            else:
                per_client_conv[key + (int(cfg["rounds"]),)] = np.asarray(v["acc_test"], dtype=float)
            rows.append({
                "rounds": int(cfg["rounds"]), "main": main_protocol,
                "qffl_L_scale": float(cfg.get("qffl_L_scale", 1.0)),
                "run_id": r.get("run_id", path.stem), "dataset": cfg["dataset"], "view": view,
                "method": cfg["method"], "aggregation": cfg["aggregation"], "qffl_q": cfg["qffl_q"],
                "alpha": cfg["dirichlet_alpha"], "eps": eps_value(priv["epsilon_target"]), "seed": cfg["seed"],
                "eps_spent": priv["epsilon_spent"], "sigma": priv["sigma"],
                **{k: s.get(k) for k in ("mean", "worst10", "std", "var", "jain", "min", "max")},
                "mia_auc": mia.get("auc_mean"), "mia_acc": mia.get("acc_mean"), "mia_adv": mia.get("adv_mean"),
                "comm_total_MB": v["comm"]["total_MB"], "comm_MB_per_round": v["comm"]["MB_per_round"],
                "rounds_comm": v["comm"]["rounds"],
                "time_per_round_s": cost["time_per_round_s"], "time_total_s": cost["time_total_s"],
                "peak_gpu_MB": cost["peak_gpu_MB"],
                "dp_gap": grp.get("pooled", {}).get("dp_gap"), "eo_gap": grp.get("pooled", {}).get("eo_gap"),
                "dp_gap_mean": grp.get("dp_gap_mean"), "eo_gap_mean": grp.get("eo_gap_mean"),
            })
    df = pd.DataFrame(rows)
    df.attrs["per_client_conv"] = per_client_conv
    return df, per_client


def add_gains(df: pd.DataFrame, per_client: dict) -> pd.DataFrame:
    """Personalisation gain per client = personalised accuracy - DP-FedAvg global accuracy (same eps, seed, data,
    and the same number of rounds)."""
    conv = df.attrs.get("per_client_conv", {})
    g_mean, g_w10, hurt = [], [], []
    for _, row in df.iterrows():
        if row.main:
            base = per_client.get((row.dataset, "fedavg", row.eps, row.seed, row.alpha))
            mine = per_client.get((row.dataset, row.view, row.eps, row.seed, row.alpha))
        else:
            base = conv.get((row.dataset, "fedavg", row.eps, row.seed, row.alpha, row.rounds))
            mine = conv.get((row.dataset, row.view, row.eps, row.seed, row.alpha, row.rounds))
        if base is None or mine is None or len(base) != len(mine):
            g_mean.append(np.nan); g_w10.append(np.nan); hurt.append(np.nan)
            continue
        gain = mine - base
        k = max(1, int(math.ceil(0.1 * len(gain))))
        g_mean.append(float(gain.mean()))
        g_w10.append(float(np.sort(mine)[:k].mean() - np.sort(base)[:k].mean()))
        hurt.append(float((gain < 0).mean()))
    df["gain_mean"], df["gain_worst10"], df["frac_hurt"] = g_mean, g_w10, hurt
    return df


def write_convergence(dfc: pd.DataFrame) -> None:
    """Runs outside the main protocol (longer non-private training): per-run rows, mean/SD over seeds, and the
    validation curves of every non-private run (both round counts) to show where training plateaus."""
    if dfc.empty:
        return
    dfc.to_csv(RESULTS / "summary_runs_converged.csv", index=False)
    keys = ["dataset", "alpha", "view", "eps", "rounds"]
    present = [m for m in METRICS if m in dfc.columns]
    agg = dfc.groupby(keys)[present].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = dfc.groupby(keys)["seed"].nunique()
    agg.reset_index().to_csv(RESULTS / "summary_converged.csv", index=False)
    curves = []
    for path in sorted(RUNS.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["privacy"]["epsilon_target"] is not None:
            continue
        c, cfg = r["curves"], r["config"]
        for rnd, m, w in zip(c["round"], c["val_mean"], c["val_worst10"]):
            curves.append({"dataset": cfg["dataset"], "method": cfg["method"], "aggregation": cfg["aggregation"],
                           "qffl_q": cfg["qffl_q"], "alpha": cfg["dirichlet_alpha"], "rounds": int(cfg["rounds"]),
                           "seed": cfg["seed"], "round": rnd, "val_mean": m, "val_worst10": w})
    pd.DataFrame(curves).to_csv(RESULTS / "curves_nonprivate.csv", index=False)
    print(f"wrote summary_converged.csv ({len(agg)} rows), curves_nonprivate.csv")


def main():
    df, per_client = load_rows()
    if df.empty:
        raise SystemExit("no finished runs in results/runs")
    df = add_gains(df, per_client)
    df = df.sort_values(["dataset", "alpha", "view", "eps", "rounds", "seed"])
    write_convergence(df[~df["main"]])
    df = df[df["main"]]
    df.to_csv(RESULTS / "summary_runs.csv", index=False)

    keys = ["dataset", "alpha", "view", "eps"]
    present = [m for m in METRICS if m in df.columns]
    agg = df.groupby(keys)[present].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby(keys)["seed"].nunique()
    agg = agg.reset_index()
    agg.to_csv(RESULTS / "summary_table.csv", index=False)

    # accuracy loss per unit epsilon
    slopes = []
    for (ds, alpha, view), g in agg.groupby(["dataset", "alpha", "view"]):
        g = g.sort_values("eps", ascending=False)
        inf_row = g[np.isinf(g.eps)]
        acc_inf = float(inf_row["mean_mean"].iloc[0]) if len(inf_row) else np.nan
        fin = g[~np.isinf(g.eps)]
        for (_, a), (_, b) in zip(fin.iloc[:-1].iterrows(), fin.iloc[1:].iterrows()):
            slopes.append({"dataset": ds, "alpha": alpha, "view": view, "eps_from": a.eps, "eps_to": b.eps,
                           "acc_from": a.mean_mean, "acc_to": b.mean_mean,
                           "loss_pts_per_unit_eps": 100 * (a.mean_mean - b.mean_mean) / (a.eps - b.eps)})
        for _, a in fin.iterrows():
            slopes.append({"dataset": ds, "alpha": alpha, "view": view, "eps_from": math.inf, "eps_to": a.eps,
                           "acc_from": acc_inf, "acc_to": a.mean_mean,
                           "drop_vs_nonprivate_pts": 100 * (acc_inf - a.mean_mean)})
    pd.DataFrame(slopes).to_csv(RESULTS / "eps_slopes.csv", index=False)

    # recommended operating configuration (Eq. 5): max mean gain subject to J >= J0 (DP-FedAvg)
    rec = []
    for (ds, alpha, eps), g in agg.groupby(["dataset", "alpha", "eps"]):
        base = g[g.view == "fedavg"]
        if base.empty:
            continue
        j0 = float(base["jain_mean"].iloc[0])
        cand = g[g.view.isin(PERSONALISED)].copy()
        feasible = cand[cand["jain_mean"] >= j0]
        pick = feasible.sort_values("gain_mean_mean", ascending=False).head(1)
        rec.append({"dataset": ds, "alpha": alpha, "eps": eps, "J0_fedavg": j0,
                    "fedavg_mean": float(base["mean_mean"].iloc[0]),
                    "recommended": pick.view.iloc[0] if len(pick) else "none (keep DP-FedAvg)",
                    "gain_pts": 100 * float(pick["gain_mean_mean"].iloc[0]) if len(pick) else 0.0,
                    "jain": float(pick["jain_mean"].iloc[0]) if len(pick) else j0,
                    "feasible": ", ".join(feasible.view.tolist())})
    pd.DataFrame(rec).to_csv(RESULTS / "recommended.csv", index=False)

    # Gate 3 sanity checks
    lines = []
    for (ds, alpha, view), g in agg.groupby(["dataset", "alpha", "view"]):
        inf_row = g[np.isinf(g.eps)]
        if len(inf_row):
            worse = g[(~np.isinf(g.eps)) & (g.mean_mean > float(inf_row.mean_mean.iloc[0]) + 0.02)]
            if len(worse):
                lines.append(f"CHECK {ds}/{view}/a{alpha}: private beats non-private by >2 pts at eps {worse.eps.tolist()}")
    big_sd = agg[agg["mean_std"] > 0.05]
    for _, r in big_sd.iterrows():
        lines.append(f"CHECK {r.dataset}/{r.view}/eps{r.eps}: seed SD of mean accuracy = {r.mean_std:.3f} (> 5 pts)")
    small = agg[(~np.isinf(agg.eps)) & (agg.eps <= 1) & (agg["mia_auc_mean"] > 0.6)]
    for _, r in small.iterrows():
        lines.append(f"CHECK {r.dataset}/{r.view}/eps{r.eps}: MIA AUC {r.mia_auc_mean:.3f} high for a small budget")
    if not lines:
        lines.append("Gate 3: no anomalies found")
    (RESULTS / "sanity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote summary_runs.csv ({len(df)} rows), summary_table.csv ({len(agg)} rows), "
          f"eps_slopes.csv, recommended.csv, sanity.txt")


if __name__ == "__main__":
    main()
