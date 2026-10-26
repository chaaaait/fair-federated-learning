"""Collect results/runs/*.json into tables (CSV) used by figures.py and the paper.

  .venv\\Scripts\\python scripts\\aggregate.py

Outputs (results/):
  summary_runs.csv        one row per (run, view) of the main protocol = configuration x epsilon x seed
  summary_table.csv       mean and SD over seeds for every metric (main protocol; n_seeds = number of seeds)
  eps_slopes.csv          accuracy loss per unit epsilon between neighbouring budgets, and drop vs non-private
  recommended.csv         problem statement Eq. (5): per epsilon, the mechanism with the largest mean
                          personalisation gain whose Jain index is >= DP-FedAvg's (the fairness floor J0)
  summary_runs_aux.csv    runs outside the main protocol, tagged by `variant`: R200 = 200 rounds, clip = clipping
  summary_aux.csv         only, C8 = clipping norm 8, lr0.1 = local learning rate, ms20 = Per-FedAvg meta-steps, k2 = IFCA clusters,
                          lam1 = an MR-MTL lambda that validation did not pick; and their mean/SD over seeds
  summary_converged.csv   the runs that differ from the main protocol only in their round count
  curves.csv              validation curves of every run (curves_nonprivate.csv: the non-private ones without clipping)
  mrmtl_lambda.csv        MR-MTL: validation and test accuracy of every lambda on the selection seeds, and the pick
  paired_eps0.csv         paired differences between mechanisms at eps0 over common seeds, with 95% t-intervals
  cost_decomposition.csv  no DP (long, main rounds) -> clipping only -> DP at eps0 (main and longer rounds)
  lr_check.csv            learning-rate check at eps0: validation picks the rate, test is reported
  sanity.txt              Gate 3 checks and run counts

The main protocol is the base configuration (configs/base.yaml) with any method, aggregation, q-FFL q and L,
Dirichlet alpha, epsilon and seed; MR-MTL enters it with the lambda chosen on validation (scripts/_common.py,
resolve_auto_lambda). Test accuracy is never used to choose anything.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from _common import BASE_YAML, QUEUE_YAML, RESULTS, RUNS, resolve_auto_lambda
from pfl.config import load_yaml, make_config

PERSONALISED = ["local", "finetune", "perfedavg", "ifca", "fedper", "mrmtl"]
METRICS = ["mean", "worst10", "std", "var", "jain", "min", "gain_mean", "gain_worst10", "frac_hurt",
           "mia_auc", "mia_acc", "mia_adv", "comm_total_MB", "comm_MB_per_round", "time_per_round_s",
           "time_total_s", "peak_gpu_MB", "dp_gap", "eo_gap", "dp_gap_mean", "eo_gap_mean",
           "eps_spent", "sigma", "val_final"]
# variant parts that change the training protocol itself; gains are then measured against DP-FedAvg run the same way
PROTOCOL_PARTS = ("R", "clip", "C", "lr")


def eps_value(e):
    return math.inf if e is None else float(e)


_DEFAULTS: dict = {}


def defaults(dataset: str) -> dict:
    """Configuration of the main protocol (configs/base.yaml) for a dataset."""
    if dataset not in _DEFAULTS:
        _DEFAULTS[dataset] = make_config({"dataset": dataset}, base_yaml=BASE_YAML)
    return _DEFAULTS[dataset]


def default_rounds(dataset: str) -> int:
    return int(defaults(dataset)["rounds"])


def variant_of(cfg: dict) -> str:
    """'' for the main protocol, otherwise the settings that differ from it, e.g. 'R200', 'clip', 'lr0.1', 'ms20',
    'k2', 'lam0.1' (MR-MTL always carries its lambda here; the chosen one is moved into the main tables)."""
    d = defaults(cfg["dataset"])
    parts = []
    if int(cfg["rounds"]) != int(d["rounds"]):
        parts.append(f"R{int(cfg['rounds'])}")
    if cfg.get("clip_only"):
        parts.append("clip")
    if float(cfg["clip_norm"]) != float(d["clip_norm"]):
        parts.append(f"C{float(cfg['clip_norm']):g}")
    if float(cfg["lr_local"]) != float(d["lr_local"]):
        parts.append(f"lr{float(cfg['lr_local']):g}")
    if cfg["method"] == "perfedavg" and cfg.get("perfedavg_meta_steps"):
        parts.append(f"ms{int(cfg['perfedavg_meta_steps'])}")
    if cfg["method"] == "ifca" and int(cfg["ifca_k"]) != int(d["ifca_k"]):
        parts.append(f"k{int(cfg['ifca_k'])}")
    if cfg["method"] == "mrmtl":
        parts.append(f"lam{float(cfg['mrmtl_lambda']):g}")
    return "-".join(parts)


_CHOICE: dict = {}


def mrmtl_choice(dataset: str, alpha: float, eps: float):
    """Lambda chosen on validation for MR-MTL (main protocol), or None while its grid is incomplete."""
    key = (dataset, alpha, eps)
    if key not in _CHOICE:
        q = load_yaml(QUEUE_YAML)
        probe = {"method": "mrmtl"}
        if alpha != float(defaults(dataset)["dirichlet_alpha"]):
            probe["dirichlet_alpha"] = alpha
        _CHOICE[key] = resolve_auto_lambda(probe, dataset, None if math.isinf(eps) else eps,
                                           q.get("mrmtl_grid", []), q.get("mrmtl_select_seeds", [0]))
    return _CHOICE[key]


def iter_runs():
    for path in sorted(RUNS.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        yield path, json.loads(path.read_text(encoding="utf-8"))


def load_rows() -> tuple[pd.DataFrame, dict]:
    """Rows for every (run, view). per_client holds the main protocol only; every other run is in
    df.attrs["per_client_aux"], keyed the same way plus its variant."""
    rows, per_client, per_client_aux = [], {}, {}
    for path, r in iter_runs():
        cfg, priv, cost = r["config"], r["privacy"], r["cost"]
        variant = variant_of(cfg)
        eps = eps_value(priv["epsilon_target"])
        alpha = float(cfg["dirichlet_alpha"])
        if cfg["method"] == "mrmtl" and variant == f"lam{float(cfg['mrmtl_lambda']):g}":
            lam = mrmtl_choice(cfg["dataset"], alpha, eps)
            main = lam is not None and float(cfg["mrmtl_lambda"]) == lam
        else:
            main = variant == ""
        vals = r["curves"]["val_mean"]
        for view, v in r["views"].items():
            s = v["summary"]
            mia = v.get("mia", {})
            grp = v.get("group", {})
            key = (cfg["dataset"], view, eps, cfg["seed"], alpha)
            if main:
                per_client[key] = np.asarray(v["acc_test"], dtype=float)
            else:
                per_client_aux[key + (variant,)] = np.asarray(v["acc_test"], dtype=float)
            rows.append({
                "rounds": int(cfg["rounds"]), "main": main, "variant": "" if main else variant,
                "qffl_L_scale": float(cfg.get("qffl_L_scale", 1.0)),
                "mrmtl_lambda": float(cfg["mrmtl_lambda"]) if cfg["method"] == "mrmtl" else np.nan,
                "lr_local": float(cfg["lr_local"]), "clip_only": bool(cfg.get("clip_only", False)),
                "run_id": r.get("run_id", path.stem), "dataset": cfg["dataset"], "view": view,
                "method": cfg["method"], "aggregation": cfg["aggregation"], "qffl_q": cfg["qffl_q"],
                "alpha": alpha, "eps": eps, "seed": cfg["seed"],
                "eps_spent": priv["epsilon_spent"], "sigma": priv["sigma"],
                **{k: s.get(k) for k in ("mean", "worst10", "std", "var", "jain", "min", "max")},
                # final validation accuracy of the model the method tracks while training (the global model for
                # FedAvg, fine-tuning and Per-FedAvg runs); used only for model selection
                "val_final": float(vals[-1]) if vals else np.nan,
                "mia_auc": mia.get("auc_mean"), "mia_acc": mia.get("acc_mean"), "mia_adv": mia.get("adv_mean"),
                "comm_total_MB": v["comm"]["total_MB"], "comm_MB_per_round": v["comm"]["MB_per_round"],
                "rounds_comm": v["comm"]["rounds"],
                "time_per_round_s": cost["time_per_round_s"], "time_total_s": cost["time_total_s"],
                "peak_gpu_MB": cost["peak_gpu_MB"],
                "dp_gap": grp.get("pooled", {}).get("dp_gap"), "eo_gap": grp.get("pooled", {}).get("eo_gap"),
                "dp_gap_mean": grp.get("dp_gap_mean"), "eo_gap_mean": grp.get("eo_gap_mean"),
            })
    df = pd.DataFrame(rows)
    df.attrs["per_client_aux"] = per_client_aux
    return df, per_client


def add_gains(df: pd.DataFrame, per_client: dict) -> pd.DataFrame:
    """Personalisation gain per client = personalised accuracy - DP-FedAvg global accuracy (same eps, seed, data).
    Runs with another protocol (rounds, clipping only, learning rate) are compared with DP-FedAvg run the same way;
    method-only settings (meta-steps, clusters, lambda) with the main DP-FedAvg."""
    aux = df.attrs.get("per_client_aux", {})
    g_mean, g_w10, hurt = [], [], []
    for _, row in df.iterrows():
        if row.main:
            base = per_client.get((row.dataset, "fedavg", row.eps, row.seed, row.alpha))
            mine = per_client.get((row.dataset, row.view, row.eps, row.seed, row.alpha))
        else:
            mine = aux.get((row.dataset, row.view, row.eps, row.seed, row.alpha, row.variant))
            base = aux.get((row.dataset, "fedavg", row.eps, row.seed, row.alpha, row.variant))
            if base is None and not any(p.startswith(PROTOCOL_PARTS) for p in row.variant.split("-")):
                base = per_client.get((row.dataset, "fedavg", row.eps, row.seed, row.alpha))
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


def mean_sd(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    present = [m for m in METRICS if m in df.columns]
    agg = df.groupby(keys, dropna=False)[present].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby(keys, dropna=False)["seed"].nunique()
    return agg.reset_index()


def write_aux(dfa: pd.DataFrame) -> None:
    """Runs outside the main protocol: per-run rows and mean/SD over seeds per variant; the subset that only
    trains for longer also goes to summary_converged.csv (used for the hollow markers in figures.py)."""
    if dfa.empty:
        return
    dfa.to_csv(RESULTS / "summary_runs_aux.csv", index=False)
    mean_sd(dfa, ["dataset", "alpha", "view", "eps", "variant", "rounds"]).to_csv(RESULTS / "summary_aux.csv",
                                                                                 index=False)
    conv = dfa[dfa.variant.str.fullmatch(r"R\d+")]
    conv.to_csv(RESULTS / "summary_runs_converged.csv", index=False)
    agg = mean_sd(conv, ["dataset", "alpha", "view", "eps", "rounds"])
    agg.to_csv(RESULTS / "summary_converged.csv", index=False)
    print(f"wrote summary_aux.csv ({dfa.groupby(['dataset', 'alpha', 'view', 'eps', 'variant']).ngroups} rows), "
          f"summary_converged.csv ({len(agg)} rows)")


def write_curves() -> None:
    curves = []
    for _, r in iter_runs():
        c, cfg = r["curves"], r["config"]
        for rnd, m, w in zip(c["round"], c["val_mean"], c["val_worst10"]):
            curves.append({"dataset": cfg["dataset"], "method": cfg["method"], "aggregation": cfg["aggregation"],
                           "qffl_q": cfg["qffl_q"], "alpha": cfg["dirichlet_alpha"], "rounds": int(cfg["rounds"]),
                           "eps": eps_value(r["privacy"]["epsilon_target"]), "variant": variant_of(cfg),
                           "seed": cfg["seed"], "round": rnd, "val_mean": m, "val_worst10": w})
    c = pd.DataFrame(curves)
    c.to_csv(RESULTS / "curves.csv", index=False)
    nonpriv = c[np.isinf(c.eps) & ~c.variant.str.contains("clip") & ~c.variant.str.contains("lam")]
    nonpriv.drop(columns=["eps", "variant"]).to_csv(RESULTS / "curves_nonprivate.csv", index=False)
    print(f"wrote curves.csv ({len(c)} rows), curves_nonprivate.csv")


def write_mrmtl(df_all: pd.DataFrame) -> None:
    """Every lambda of the MR-MTL grid on the selection seeds (validation picks, test is only reported) next to
    local training (lambda = 0) on the same seeds."""
    q = load_yaml(QUEUE_YAML)
    sel = q.get("mrmtl_select_seeds", [0])
    m = df_all[(df_all.method == "mrmtl") & (df_all.rounds == df_all.dataset.map(default_rounds))
               & df_all.seed.isin(sel)]
    if m.empty:
        return
    rows = []
    for (ds, alpha, eps), g in m.groupby(["dataset", "alpha", "eps"]):
        lam_star = mrmtl_choice(ds, alpha, eps)
        loc = df_all[(df_all.dataset == ds) & (df_all.alpha == alpha) & (df_all.eps == eps) & (df_all.view == "local")
                     & df_all.main & df_all.seed.isin(sel)]
        if len(loc):
            rows.append({"dataset": ds, "alpha": alpha, "eps": eps, "lambda": 0.0, "is_local": True,
                         "val_final": loc.val_final.mean(), "test_mean": loc["mean"].mean(),
                         "test_worst10": loc.worst10.mean(), "chosen": False})
        for lam, h in g.groupby("mrmtl_lambda"):
            rows.append({"dataset": ds, "alpha": alpha, "eps": eps, "lambda": lam, "is_local": False,
                         "val_final": h.val_final.mean(), "test_mean": h["mean"].mean(),
                         "test_worst10": h.worst10.mean(), "chosen": lam_star is not None and lam == lam_star})
    pd.DataFrame(rows).to_csv(RESULTS / "mrmtl_lambda.csv", index=False)
    print("wrote mrmtl_lambda.csv")


def write_paired(df: pd.DataFrame, eps0: float) -> None:
    """Paired differences between two mechanisms over the seeds both have at eps0 (same data, same seed), per
    dataset on its main partition; 95% interval = mean +- t(0.975, n-1) * SD / sqrt(n)."""
    rows = []
    for dataset in sorted(df.dataset.unique()):
        alpha = float(defaults(dataset)["dirichlet_alpha"])
        m = df[(df.dataset == dataset) & (df.alpha == alpha) & (df.eps == eps0)]
        views = sorted(set(m.view))
        pairs = [(v, "fedavg") for v in views if v != "fedavg"]
        pairs += [("local", v) for v in ("fedper", "mrmtl", "finetune", "perfedavg", "ifca") if v in views]
        for a, b in pairs:
            A, B = m[m.view == a].set_index("seed"), m[m.view == b].set_index("seed")
            seeds = sorted(set(A.index) & set(B.index))
            if len(seeds) < 2:
                continue
            for metric in ("mean", "worst10", "jain"):
                d = (A.loc[seeds, metric] - B.loc[seeds, metric]).to_numpy(dtype=float)
                n, mu, sd = len(d), float(d.mean()), float(d.std(ddof=1))
                half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
                rows.append({"dataset": dataset, "alpha": alpha, "eps": eps0, "a": a, "b": b, "metric": metric,
                             "n_seeds": n, "seeds": " ".join(map(str, seeds)), "diff_mean": mu, "diff_sd": sd,
                             "ci95_low": mu - half, "ci95_high": mu + half,
                             "a_wins_all_seeds": bool((d > 0).all())})
    pd.DataFrame(rows).to_csv(RESULTS / "paired_eps0.csv", index=False)
    print(f"wrote paired_eps0.csv ({len(rows)} rows)")


def write_decomposition(df_all: pd.DataFrame, eps0: float) -> None:
    """Cost of DP in steps, per mechanism: longer non-private training (compute), clipping without noise, and noise,
    plus what more private rounds buy back. Each cell is a mean over seeds (n given)."""
    rows = []
    for ds in sorted(df_all.dataset.unique()):
        alpha = float(defaults(ds)["dirichlet_alpha"])
        d = df_all[(df_all.dataset == ds) & (df_all.alpha == alpha)]
        if d.empty:
            continue
        R = default_rounds(ds)
        long_R = int(d[np.isinf(d.eps) & d.variant.str.fullmatch(r"R\d+")].rounds.max()) \
            if (np.isinf(d.eps) & d.variant.str.fullmatch(r"R\d+")).any() else None
        stages = {"nodp_long": (math.inf, f"R{long_R}" if long_R else None, False),
                  "nodp": (math.inf, "", True), "clip": (math.inf, "clip", False), "dp": (eps0, "", True),
                  "dp_R80": (eps0, "R80", False), "dp_R200": (eps0, "R200", False), "dp_C8": (eps0, "C8", False)}
        for view in ("fedavg", "finetune", "local", "fedper", "perfedavg", "ifca"):
            for metric in ("mean", "worst10", "jain", "dp_gap", "eo_gap"):
                if metric in ("dp_gap", "eo_gap") and d[metric].isna().all():
                    continue  # group gaps exist only for tabular data with a sensitive attribute
                row = {"dataset": ds, "alpha": alpha, "view": view, "metric": metric, "eps0": eps0,
                       "rounds_main": R, "rounds_long": long_R}
                for stage, (eps, var, main) in stages.items():
                    if var is None:
                        continue
                    sel = d[(d.view == view) & ((np.isinf(d.eps)) if math.isinf(eps) else (d.eps == eps))]
                    sel = sel[sel.main] if main else sel[(~sel.main) & (sel.variant == var)]
                    row[f"{stage}_mean"] = sel[metric].mean() if len(sel) else np.nan
                    row[f"{stage}_sd"] = sel[metric].std() if len(sel) > 1 else np.nan
                    row[f"{stage}_n"] = int(sel.seed.nunique())
                g = lambda k: row.get(f"{k}_mean", np.nan)  # noqa: E731
                row.update({"compute": g("nodp_long") - g("nodp"), "clipping": g("nodp") - g("clip"),
                            "noise": g("clip") - g("dp"), "total_vs_long": g("nodp_long") - g("dp"),
                            "total_equal_rounds": g("nodp") - g("dp"),
                            "dp_more_rounds_R80": g("dp_R80") - g("dp"), "dp_more_rounds_R200": g("dp_R200") - g("dp"),
                            "dp_clip8_vs_clip1": g("dp_C8") - g("dp"),
                            "cost_at_R200": g("nodp_long") - g("dp_R200") if long_R == 200 else np.nan})
                rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "cost_decomposition.csv", index=False)
    print("wrote cost_decomposition.csv")


def write_lr_check(df_all: pd.DataFrame, eps0: float) -> None:
    """Learning-rate check at eps0: the rate with the best final validation accuracy is picked per mechanism
    (fine-tuning shares the FedAvg run, so it follows FedAvg's pick); test accuracy is reported for every rate."""
    tables = [_lr_check(df_all, eps0, ds, float(defaults(ds)["dirichlet_alpha"])) for ds in sorted(df_all.dataset.unique())]
    tables = [t for t in tables if t is not None]
    if tables:
        pd.concat(tables).to_csv(RESULTS / "lr_check.csv", index=False)
        print("wrote lr_check.csv")


def _lr_check(df_all: pd.DataFrame, eps0: float, dataset: str, alpha: float):
    d = df_all[(df_all.dataset == dataset) & (df_all.alpha == alpha) & (df_all.eps == eps0)
               & ((df_all.main) | df_all.variant.str.fullmatch(r"lr[\d.]+"))]
    lr_runs = d[d.variant != ""]
    if lr_runs.empty:
        return None
    rows = []
    for view in sorted(set(lr_runs.view)):
        g = d[d.view == view]
        common = set.intersection(*(set(h.seed) for _, h in g.groupby("lr_local")))  # like with like
        g = g[g.seed.isin(common)]
        for lr, h in g.groupby("lr_local"):
            rows.append({"dataset": dataset, "alpha": alpha, "eps": eps0, "view": view, "lr_local": lr,
                         "n_seeds": h.seed.nunique(), "val_final": h.val_final.mean(),
                         "mean_mean": h["mean"].mean(), "mean_std": h["mean"].std(),
                         "worst10_mean": h.worst10.mean(), "worst10_std": h.worst10.std(),
                         "jain_mean": h.jain.mean()})
    out = pd.DataFrame(rows)
    out["chosen_on_val"] = False
    for view, g in out.groupby("view"):
        src = "fedavg" if view == "finetune" else view
        s = out[out.view == src]
        if len(s):
            best = s.loc[s.val_final.idxmax(), "lr_local"]
            out.loc[g.index, "chosen_on_val"] = g.lr_local == best
    return out


def run_counts(df_all: pd.DataFrame) -> list[str]:
    runs = df_all.drop_duplicates("run_id")
    priv = runs[~np.isinf(runs.eps)]
    excess = (priv.eps_spent - priv.eps).max() if len(priv) else float("nan")
    over = int((priv.eps_spent > priv.eps + 1e-9).sum())
    return [f"runs: {len(runs)} result files ({int(runs.main.sum())} main protocol, {int((~runs.main).sum())} other); "
            f"{len(priv)} private, {over} over budget (largest eps_spent - target = {excess:.2g})",
            f"GPU time: {runs.time_total_s.sum() / 3600:.1f} h in total "
            f"({runs[runs.main].time_total_s.sum() / 3600:.1f} h main protocol)"]


def main():
    df, per_client = load_rows()
    if df.empty:
        raise SystemExit("no finished runs in results/runs")
    df = add_gains(df, per_client)
    df = df.sort_values(["dataset", "alpha", "view", "eps", "variant", "rounds", "seed"])
    eps0 = float(load_yaml(QUEUE_YAML).get("eps0", 2.0))
    df_all = df
    write_aux(df[~df["main"]])
    write_curves()
    write_mrmtl(df_all)
    write_decomposition(df_all, eps0)
    write_lr_check(df_all, eps0)
    df = df[df["main"]]
    write_paired(df, eps0)
    df.to_csv(RESULTS / "summary_runs.csv", index=False)

    keys = ["dataset", "alpha", "view", "eps"]
    agg = mean_sd(df, keys)
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
    lines = run_counts(df_all)
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
    if len(lines) == 2:
        lines.append("Gate 3: no anomalies found")
    (RESULTS / "sanity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote summary_runs.csv ({len(df)} rows), summary_table.csv ({len(agg)} rows), "
          f"eps_slopes.csv, recommended.csv, sanity.txt")


if __name__ == "__main__":
    main()
