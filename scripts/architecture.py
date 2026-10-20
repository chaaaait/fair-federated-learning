"""Architecture diagram (paper Fig. 1 + deck): python scripts/architecture.py -> paper/figures/architecture.pdf|png"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PFL_PAPER_OUT", ROOT / "paper")) / "figures"

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, BLUE_BG = "#2a78d6", "#e6f0fb"
GREEN, GREEN_BG = "#008300", "#e5f3e5"
ORANGE, ORANGE_BG = "#eb6834", "#fdece4"
GREY_BG = "#f2f1ee"
TITLE_H, LINE_H, PAD = 0.21, 0.155, 0.10
FS_T, FS_B = 7.0, 6.2


def height(n_lines: int) -> float:
    return TITLE_H + n_lines * LINE_H + PAD


def box(ax, x, y_top, w, title, lines=(), edge=INK2, face="white", title_color=INK, lw=1.0, h=None):
    """Draw a rounded box whose top edge is at y_top; returns its bottom y."""
    h = height(len(lines)) if h is None else h
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                linewidth=lw, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y_top - 0.06, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y_top - TITLE_H - 0.03 - i * LINE_H, line, ha="center", va="top", fontsize=FS_B, color=INK)
    return y


def arrow(ax, p, q, color=INK2, label="", label_xy=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=1.2, color=color))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=FS_B, color=color)


def main():
    W, H = 7.16, 4.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # data partition band
    box(ax, 0.05, 3.95, W - 0.1, "Data partition (fixed for every run)",
        ["CIFAR-10: 60k images, Dirichlet(α = 0.3) label skew → 20 clients   ·   Adult: one client per occupation → 12 clients",
         "every client splits its own data 70 / 10 / 20 (train / validation / test)"], face=GREY_BG)

    top = 3.22
    # client column
    cx, cw = 0.05, 2.5
    ix, iw = cx + 0.12, cw - 0.24
    box(ax, cx, top, cw, "Client k  (× K, every round)", edge=BLUE, face=BLUE_BG, title_color=BLUE,
        lw=1.4, h=top - 0.08)
    y = box(ax, ix, top - 0.34, iw, "Private data Dₖ", ["records never leave the client"])
    y_dp_top = y - 0.12
    y = box(ax, ix, y_dp_top, iw, "DP-SGD step (Opacus)",
            ["Poisson batch, sampling rate q = 0.03", "clip each example's gradient to C = 1",
             "add Gaussian noise N(0, σ²C²)", "same queries for every client"])
    y_dp_bottom = y
    y_pm_top = y - 0.12
    y = box(ax, ix, y_pm_top, iw, "Personalisation mechanism",
            ["★ the only variable under study ★", "Fine-tune · Per-FedAvg · IFCA · FedPer",
             "baselines: DP-FedAvg · Local-only", "output: personalised model vₖ"],
            edge=ORANGE, face=ORANGE_BG, title_color=ORANGE, lw=1.5)
    y_pm_mid = (y_pm_top + y) / 2
    arrow(ax, (ix + iw / 2, y_dp_top + 0.12), (ix + iw / 2, y_dp_top))
    arrow(ax, (ix + iw / 2, y_dp_bottom), (ix + iw / 2, y_pm_top))

    # server column
    sx, sw = 3.33, 1.98
    s_inner = [("Aggregation", ["FedAvg: data-weighted mean", "q-FFL: up-weights", "high-loss clients (q = 1, 5)"]),
               ("Privacy accountant", ["Rényi DP over all queries", "σ calibrated per run", "→ every method spends ε₀"])]
    s_h = 0.34 + sum(height(len(lines)) for _, lines in s_inner) + 0.12 * (len(s_inner) - 1) + 0.12
    box(ax, sx, top, sw, "Server", edge=GREEN, face=GREEN_BG, title_color=GREEN, lw=1.4, h=s_h)
    y = top - 0.34
    for title, lines in s_inner:
        y = box(ax, sx + 0.1, y, sw - 0.2, title, lines) - 0.12

    # evaluation column
    ex, ew = 5.55, W - 5.55 - 0.05
    box(ax, ex, top, ew, "Evaluation", face=GREY_BG, lw=1.2, h=top - 0.08)
    y = top - 0.34
    for title, lines in [("Utility", ["mean accuracy", "gain over DP-FedAvg"]),
                         ("Fairness", ["worst-10% · SD · Jain", "DP / EO gaps (Adult)"]),
                         ("Privacy", ["ε, δ · MIA AUC", "loss-threshold attack"]),
                         ("Cost", ["MB / round · s / round", "peak GPU memory"])]:
        y = box(ax, ex + 0.1, y, ew - 0.2, title, lines) - 0.1

    # exchanges between client and server
    up_y, down_y = top - 0.95, top - 1.75
    arrow(ax, (cx + cw, up_y), (sx, up_y + 0.12), BLUE, "noised\nupdate", ((cx + cw + sx) / 2, up_y + 0.33))
    arrow(ax, (sx, down_y), (cx + cw, down_y - 0.12), GREEN, "new\nmodel(s)", ((cx + cw + sx) / 2, down_y - 0.36))
    # personalised models to evaluation
    arrow(ax, (ix + iw, y_pm_mid), (ex, y_pm_mid), INK2, "per-client test accuracy and losses, 3 seeds",
          ((ix + iw + ex) / 2, y_pm_mid - 0.16))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "architecture.png", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT / 'architecture.pdf'} and .png")


if __name__ == "__main__":
    main()
