"""Paper figures and LaTeX tables from results/summary_*.csv (run aggregate.py first).

  .venv\\Scripts\\python scripts\\figures.py [--eps0 2]

Writes paper/figures/*.pdf (vector, for LaTeX) + *.png (300 dpi, for slides) and paper/tables/*.tex.
Any figure whose data is missing is skipped with a note, so this also works on partial results.

Style: one fixed colour + marker per configuration (colour follows the entity, never its rank),
thin lines, error bars = SD over seeds, one y-axis per panel, recessive grid. The palette is the
dataviz reference categorical palette (validated: CVD-safe on adjacent pairs); three hues are
below 3:1 contrast, so every series also has its own marker and a legend.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS, ROOT  # noqa: E402

PAPER_OUT = Path(os.environ.get("PFL_PAPER_OUT", ROOT / "paper"))  # override for tests
FIG = PAPER_OUT / "figures"
TAB = PAPER_OUT / "tables"

# configuration -> (label, colour, marker). Order = legend order.
STYLE = {
    "fedavg":    ("DP-FedAvg (global)", "#2a78d6", "o"),
    "finetune":  ("Fine-tuning",        "#eb6834", "s"),
    "local":     ("Local only",         "#1baf7a", "^"),
    "perfedavg": ("Per-FedAvg",         "#eda100", "D"),
    "ifca":      ("Clustered (IFCA)",   "#e87ba4", "v"),
    "fedper":    ("FedPer",             "#008300", "P"),
    "qffl_q1":   ("q-FFL, q=1",         "#6250d6", "X"),
    "qffl_q5":   ("q-FFL, q=5",         "#e34948", "*"),
}
MAIN = ["fedavg", "finetune", "local", "perfedavg", "ifca", "fedper"]
SHORT = {"fedavg": "DP-FedAvg", "finetune": "Fine-tune", "local": "Local", "perfedavg": "Per-FedAvg",
         "ifca": "Clustered", "fedper": "FedPer"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
INF_POS = 16.0  # where "non-private" sits on the log epsilon axis

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.axisbelow": True, "figure.constrained_layout.use": True,
})
COL, DBL = 3.5, 7.16  # IEEE single / double column width (inches)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  figure {name}")


def xpos(eps):
    return INF_POS if np.isinf(eps) else eps


def eps_axis(ax, eps_values):
    finite = sorted(e for e in eps_values if not np.isinf(e))
    ax.set_xscale("log")
    ticks = finite + ([INF_POS] if any(np.isinf(e) for e in eps_values) else [])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{e:g}" for e in finite] + (["∞"] if len(ticks) > len(finite) else []))
    ax.minorticks_off()
    ax.set_xlabel("privacy budget ε (δ = 10⁻⁵); ∞ = no DP")


def line_panel(ax, t, views, metric, ylabel, pct=True):
    shown = False
    for v in views:
        g = t[t.view == v].sort_values("eps")
        if g.empty:
            continue
        label, color, marker = STYLE[v]
        k = 100 if pct else 1
        ax.errorbar([xpos(e) for e in g.eps], g[f"{metric}_mean"] * k, yerr=g[f"{metric}_std"].fillna(0) * k,
                    color=color, marker=marker, label=label, capsize=2, elinewidth=0.8)
        shown = True
    if shown:
        eps_axis(ax, t.eps.unique())
        ax.set_ylabel(ylabel)
    return shown


def fig_m1(t):
    views = ["fedavg", "finetune"]
    if t[t.view.isin(views)].empty:
        return print("  skip m1 (no FedAvg runs yet)")
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.4))
    line_panel(axes[0], t, views, "mean", "mean client accuracy (%)")
    line_panel(axes[1], t, views, "worst10", "worst-10% client accuracy (%)")
    axes[0].set_title("(a) average client")
    axes[1].set_title("(b) worst 10% of clients")
    axes[0].legend(frameon=False)
    save(fig, "m1_accuracy_vs_eps")

    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.4))
    line_panel(axes[0], t, views, "jain", "Jain's fairness index", pct=False)
    line_panel(axes[1], t, views, "std", "SD of client accuracy (pts)")
    axes[0].set_title("(a) evenness (1 = all clients equal)")
    axes[1].set_title("(b) spread across clients")
    axes[0].legend(frameon=False)
    save(fig, "m1_fairness_vs_eps")


def fig_m2_bars(t, eps0):
    g = t[(t.eps == eps0) & t.view.isin(MAIN)]
    if g.empty:
        return print(f"  skip m2 bars (no runs at eps0={eps0})")
    order = [v for v in MAIN if v in set(g.view)]
    g = g.set_index("view").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.3))
    specs = [("mean", "mean accuracy (%)", 100), ("worst10", "worst-10% accuracy (%)", 100), ("jain", "Jain's index", 1)]
    x = np.arange(len(order))
    for ax, (m, lab, k) in zip(axes, specs):
        colors = [STYLE[v][1] for v in order]
        mu, sd = g[f"{m}_mean"].to_numpy() * k, g[f"{m}_std"].fillna(0).to_numpy() * k
        top = 1.0 if m == "jain" else 100.0
        yerr = np.vstack([np.minimum(sd, mu), np.minimum(sd, top - mu)])  # keep bars inside [0, top]
        ax.bar(x, mu, yerr=yerr, color=colors, width=0.7,
               edgecolor="white", linewidth=1.0, capsize=2, error_kw={"elinewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[v][0].replace(" (global)", "").replace(" (IFCA)", "") for v in order],
                           rotation=35, ha="right")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        if m == "jain":
            ax.set_ylim(max(0, float((g[f'{m}_mean']).min()) - 0.1), 1.0)
    fig.suptitle(f"All mechanisms at the same budget ε = {eps0:g} (mean ± SD over seeds)", fontsize=8.5)
    save(fig, "m2_mechanisms_at_eps0")


def fig_m2_sweep(t):
    if t[t.view.isin(MAIN)].eps.nunique() < 3:
        return print("  skip m2 sweep (fewer than 3 budgets)")
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.6))
    line_panel(axes[0], t, MAIN, "mean", "mean client accuracy (%)")
    line_panel(axes[1], t, MAIN, "worst10", "worst-10% client accuracy (%)")
    axes[0].set_title("(a) average client")
    axes[1].set_title("(b) worst 10% of clients")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    save(fig, "m2_accuracy_vs_eps_all")


def fig_gain_boxes(runs, eps0, dataset="cifar10", alpha=0.3):
    """Per-client personalisation gain at eps0, pooled over seeds (needs per-client accuracies)."""
    from aggregate import load_rows  # per-client arrays live in the run JSONs

    _, per_client = load_rows()
    data, labels, colors, hurt = [], [], [], []
    for v in [x for x in MAIN if x != "fedavg"]:
        gains = []
        for (ds, view, eps, seed, alpha), acc in per_client.items():
            if ds == dataset and view == v and eps == eps0 and abs(alpha - 0.3) < 1e-9:
                base = per_client.get((ds, "fedavg", eps, seed, alpha))
                if base is not None:
                    gains.extend(((acc - base) * 100).tolist())
        if gains:
            data.append(gains)
            labels.append(SHORT.get(v, STYLE[v][0]))
            colors.append(STYLE[v][1])
            hurt.append(100 * float(np.mean(np.asarray(gains) < 0)))
    if not data:
        return print("  skip gain boxes (need FedAvg + mechanisms at eps0)")
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": INK, "linewidth": 1.0},
                    flierprops={"markersize": 2.5, "markeredgecolor": INK2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    ax.axhline(0, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{l}\n{h:.0f}% hurt" for l, h in zip(labels, hurt)], rotation=0, fontsize=6.5)
    ax.set_ylabel("gain over DP-FedAvg (pts)")
    ax.set_title(f"Per-client personalisation gain at ε = {eps0:g}")
    ax.grid(axis="x", visible=False)
    save(fig, "m2_gain_per_client")


def fig_qffl(t, eps0):
    pts = t[t.view.isin(["fedavg", "qffl_q1", "qffl_q5"]) & (t.eps.isin([eps0]) | np.isinf(t.eps))]
    if pts[pts.view != "fedavg"].empty:
        return print("  skip q-FFL (no runs yet)")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for eps, ls in ((math.inf, ":"), (eps0, "-")):
        g = pts[(np.isinf(pts.eps) if np.isinf(eps) else pts.eps == eps)]
        g = g.set_index("view").reindex(["fedavg", "qffl_q1", "qffl_q5"]).dropna(subset=["mean_mean"])
        ax.plot(g["mean_mean"] * 100, g["worst10_mean"] * 100, color=INK2, linestyle=ls, linewidth=0.9, zorder=1)
        for v, r in g.iterrows():
            label, color, marker = STYLE[v]
            ax.errorbar(r["mean_mean"] * 100, r["worst10_mean"] * 100, xerr=(r["mean_std"] or 0) * 100,
                        yerr=(r["worst10_std"] or 0) * 100, color=color, marker=marker, capsize=2,
                        markersize=6, linestyle="none",
                        label=f"{label.replace(' (global)', '').replace('DP-FedAvg', 'FedAvg (q=0)')}"
                        if np.isinf(eps) else None, zorder=2)
        tag = "no DP" if np.isinf(eps) else f"ε = {eps0:g}"
        if len(g):
            ax.annotate(tag, (g["mean_mean"].iloc[-1] * 100, g["worst10_mean"].iloc[-1] * 100),
                        textcoords="offset points", xytext=(6, 6), fontsize=7, color=INK2)
    ax.set_xlabel("mean client accuracy (%)")
    ax.set_ylabel("worst-10% client accuracy (%)")
    ax.set_title("q-FFL: worst-client gain vs mean-accuracy cost")
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=INK2, linestyle="-", linewidth=0.9),
                Line2D([], [], color=INK2, linestyle=":", linewidth=0.9)]
    labels += [f"ε = {eps0:g}", "no DP"]
    ax.legend(handles, labels, frameon=False, loc="best")
    save(fig, "m3_qffl_tradeoff")


def fig_mia(t):
    views = [v for v in MAIN if v in set(t.view)]
    g = t[t.view.isin(views) & t["mia_auc_mean"].notna()]
    if g.eps.nunique() < 2:
        return print("  skip MIA (need >= 2 budgets)")
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    line_panel(ax, g, views, "mia_auc", "membership-inference AUC", pct=False)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle="--")
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.5 - 0.15 * (hi - lo)), hi)  # room under the 0.5 line for its label
    ax.annotate("random guessing", (min(g.eps), 0.5), textcoords="offset points", xytext=(2, -3), va="top",
                fontsize=6.5, color=INK2)
    ax.set_title("Measured leakage vs theoretical budget")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save(fig, "privacy_mia_vs_eps")


# ----------------------------------------------------------------------------- tables
def fmt(m, s, k=100, d=1):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    s = 0.0 if s is None or (isinstance(s, float) and math.isnan(s)) else s
    return f"{m * k:.{d}f} $\\pm$ {s * k:.{d}f}"


def table_main(t, eps0):
    g = t[(t.eps == eps0) & t.view.isin(MAIN)]
    if g.empty:
        return print("  skip table main")
    rows = []
    for v in [x for x in MAIN if x in set(g.view)]:
        r = g[g.view == v].iloc[0]
        rows.append(" & ".join([STYLE[v][0], fmt(r.mean_mean, r.mean_std), fmt(r.worst10_mean, r.worst10_std),
                                fmt(r.std_mean, r.std_std), fmt(r.jain_mean, r.jain_std, 1, 3),
                                fmt(r.gain_mean_mean, r.gain_mean_std) if v != "fedavg" else "0 (ref.)",
                                fmt(r.mia_auc_mean, r.mia_auc_std, 1, 3),
                                f"{r.comm_total_MB_mean:.0f}"]) + r" \\")
    tex = (r"\begin{tabular}{lccccccc}" "\n\\toprule\n"
           r"Mechanism & Mean acc. (\%) & Worst-10\% (\%) & SD (pts) & Jain & Gain (pts) & MIA AUC & Comm. (MB) \\"
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "main_results.tex").write_text(tex, encoding="utf-8")
    print("  table main_results.tex")


def table_adult(t, eps0):
    a = t[t.dataset.isin(["adult", "synthetic_tab"]) & t.view.isin(MAIN)]
    if a.empty:
        return print("  skip table adult")
    rows = []
    for v in [x for x in MAIN if x in set(a.view)]:
        cells = [STYLE[v][0]]
        for eps in (math.inf, eps0):
            r = a[(a.view == v) & ((np.isinf(a.eps)) if np.isinf(eps) else (a.eps == eps))]
            if r.empty:
                cells += ["--", "--", "--"]
                continue
            r = r.iloc[0]
            cells += [fmt(r.mean_mean, r.mean_std), fmt(r.dp_gap_mean, r.dp_gap_std, 100, 1),
                      fmt(r.eo_gap_mean, r.eo_gap_std, 100, 1)]
        rows.append(" & ".join(cells) + r" \\")
    tex = (r"\begin{tabular}{lcccccc}" "\n\\toprule\n"
           r" & \multicolumn{3}{c}{No DP} & \multicolumn{3}{c}{$\varepsilon=" + f"{eps0:g}" + r"$} \\" "\n"
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
           r"Mechanism & Acc. (\%) & DP gap (pts) & EO gap (pts) & Acc. (\%) & DP gap (pts) & EO gap (pts) \\"
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "adult_group_fairness.tex").write_text(tex, encoding="utf-8")
    print("  table adult_group_fairness.tex")


def table_cost(t, eps0):
    g = t[t.view.isin(MAIN + ["qffl_q1"])]
    if g.empty:
        return print("  skip table cost")
    rows = []
    for v in [x for x in MAIN + ["qffl_q1"] if x in set(g.view)]:
        priv = g[(g.view == v) & (g.eps == eps0)]
        non = g[(g.view == v) & np.isinf(g.eps)]
        if priv.empty:
            continue
        p = priv.iloc[0]
        tp, tn = p.time_per_round_s_mean, (non.iloc[0].time_per_round_s_mean if len(non) else float("nan"))
        ratio = tp / tn if tn and not math.isnan(tn) else float("nan")
        rows.append(" & ".join([STYLE[v][0], f"{p.comm_MB_per_round_mean:.1f}", f"{p.comm_total_MB_mean:.0f}",
                                f"{tn:.1f}" if not math.isnan(tn) else "--", f"{tp:.1f}",
                                f"{ratio:.1f}$\\times$" if not math.isnan(ratio) else "--",
                                f"{p.peak_gpu_MB_mean:.0f}" if not math.isnan(p.peak_gpu_MB_mean or float('nan')) else "--"])
                    + r" \\")
    tex = (r"\begin{tabular}{lcccccc}" "\n\\toprule\n"
           r"Mechanism & MB/round & Total MB & s/round (no DP) & s/round (DP) & DP overhead & Peak GPU (MB) \\"
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "cost.tex").write_text(tex, encoding="utf-8")
    print("  table cost.tex")


def table_recommended(dataset="cifar10"):
    p = RESULTS / "recommended.csv"
    if not p.exists():
        return print("  skip table recommended")
    rec = pd.read_csv(p)
    rec = rec[rec.dataset == dataset]
    rec = rec[rec.alpha == rec.alpha.min()].sort_values("eps", ascending=False)  # main partition; alpha=1 check is in the text
    rows = []
    for _, r in rec.iterrows():
        eps = "$\\infty$" if np.isinf(r.eps) else f"{r.eps:g}"
        name = STYLE.get(r.recommended, (r.recommended,))[0]
        rows.append(f"{eps} & {r.fedavg_mean * 100:.1f} & {r.J0_fedavg:.3f} & {name} & {r.gain_pts:+.1f} & {r.jain:.3f}" + r" \\")
    tex = (r"\begin{tabular}{cccccc}" "\n\\toprule\n"
           r"$\varepsilon$ & DP-FedAvg acc. (\%) & $J_0$ & Recommended & Gain (pts) & Jain \\"
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "recommended.tex").write_text(tex, encoding="utf-8")
    print("  table recommended.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps0", type=float, default=None, help="common budget (default: eps0 in configs/queue.yaml)")
    ap.add_argument("--dataset", default="cifar10")
    args = ap.parse_args()
    if args.eps0 is None:
        from pfl.config import load_yaml

        from _common import QUEUE_YAML

        args.eps0 = float(load_yaml(QUEUE_YAML).get("eps0", 2.0))
    path = RESULTS / "summary_table.csv"
    if not path.exists():
        raise SystemExit("run scripts/aggregate.py first")
    t_all = pd.read_csv(path)
    t = t_all[(t_all.dataset == args.dataset) & (t_all.alpha == 0.3)]
    print(f"figures (eps0 = {args.eps0:g}):")
    fig_m1(t)
    fig_m2_bars(t, args.eps0)
    fig_m2_sweep(t)
    fig_gain_boxes(t, args.eps0, args.dataset)
    fig_qffl(t, args.eps0)
    fig_mia(t)
    print("tables:")
    table_main(t, args.eps0)
    table_adult(t_all, args.eps0)
    table_cost(t, args.eps0)
    table_recommended(args.dataset)


if __name__ == "__main__":
    main()
