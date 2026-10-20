"""Dataset description for the rubric's Dataset criterion and paper Table T2.

  .venv\\Scripts\\python scripts\\dataset_stats.py            # CIFAR-10 + Adult (downloads on first run)
  .venv\\Scripts\\python scripts\\dataset_stats.py --dataset synthetic

Writes results/dataset_stats.json, paper/tables/datasets.tex and paper/figures/partition_<dataset>.pdf/png
(stacked bar per client: how many examples of each class -> shows the label skew).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _common import BASE_YAML, RESULTS, ROOT  # noqa: E402
from pfl.config import make_config  # noqa: E402
from pfl.data import CIFAR_CLASSES, build_federated_data  # noqa: E402
from pfl.utils import write_json_atomic  # noqa: E402

PAPER_OUT = Path(os.environ.get("PFL_PAPER_OUT", ROOT / "paper"))
# 10 classes need a sequential-free categorical set; reuse the validated 8 + 2 neutral greys.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
                "#8a8984", "#c3c2b7"]

SOURCES = {
    "cifar10": ("CIFAR-10", "Krizhevsky (2009), Univ. of Toronto", "No formal licence; free research use with citation",
                "32x32 RGB images, 3,072 values"),
    "adult": ("Adult (Census Income)", "UCI ML Repository, Becker & Kohavi (1996)", "CC BY 4.0",
              "14 attributes -> 5 numeric + one-hot categorical"),
    "synthetic": ("Synthetic images", "generated (tests only)", "n/a", "16x16 RGB"),
    "synthetic_tab": ("Synthetic tabular", "generated (tests only)", "n/a", "12 numeric"),
}


def describe(name: str) -> dict:
    cfg = make_config({"dataset": name, "device": "cpu"}, base_yaml=BASE_YAML)
    data = build_federated_data(cfg, torch.device("cpu"))
    m = data.meta
    sizes = np.array([c["n"] for c in m["clients"]])
    hist = np.array([c["class_hist"] for c in m["clients"]])
    total_hist = hist.sum(0)
    out = {
        "dataset": name, "n_total": m["n_total"], "num_clients": m["num_clients"], "num_classes": m["num_classes"],
        "num_features": m["num_features"], "client_size_min": int(sizes.min()),
        "client_size_median": int(np.median(sizes)), "client_size_max": int(sizes.max()),
        "class_counts_total": total_hist.tolist(),
        "classes_per_client_over_5pct": [int((h / h.sum() > 0.05).sum()) for h in hist],
        "split": cfg["split"], "dirichlet_alpha": cfg["dirichlet_alpha"] if data.image else None,
        "clients": m["clients"],
    }
    # stacked bar chart of class counts per client
    fig, ax = plt.subplots(figsize=(7.16, 2.2), layout="constrained")
    bottom = np.zeros(len(hist))
    labels = CIFAR_CLASSES if name == "cifar10" else (["<=50K", ">50K"] if not data.image else
                                                     [f"class {i}" for i in range(hist.shape[1])])
    for j in range(hist.shape[1]):
        ax.bar(np.arange(len(hist)), hist[:, j], bottom=bottom, color=CLASS_COLORS[j % len(CLASS_COLORS)],
               edgecolor="white", linewidth=0.6, label=labels[j], width=0.8)
        bottom += hist[:, j]
    ax.set_xticks(np.arange(len(hist)))
    names = [c["name"] for c in m["clients"]]
    ax.set_xticklabels(names if not data.image else [str(i) for i in range(len(hist))],
                       rotation=45 if not data.image else 0, ha="right" if not data.image else "center", fontsize=6.5)
    ax.set_xlabel("client" if data.image else "client (occupation)")
    ax.set_ylabel("examples")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    fig.legend(ncol=min(10, hist.shape[1]), fontsize=6, frameon=False, loc="outside upper center")
    figdir = PAPER_OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"partition_{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"partition_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if not data.image:
        out["female_frac_per_client"] = [c.get("female_frac") for c in m["clients"]]
        out["positive_rate_per_client"] = [c.get("positive_rate") for c in m["clients"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["cifar10", "adult"])
    args = ap.parse_args()
    stats = {name: describe(name) for name in args.dataset}
    write_json_atomic(RESULTS / "dataset_stats.json", stats)
    names = list(stats)
    head = " & ".join(["", *[SOURCES[n][0] for n in names]]) + r" \\"
    def row(label, fn):
        return " & ".join([label, *[fn(n, stats[n]) for n in names]]) + r" \\"
    rows = [
        row("Source", lambda n, s: SOURCES[n][1]),
        row("Licence", lambda n, s: SOURCES[n][2]),
        row("Records", lambda n, s: f"{s['n_total']:,}"),
        row("Features", lambda n, s: SOURCES[n][3]),
        row("Classes", lambda n, s: str(s["num_classes"])),
        row("Clients", lambda n, s: str(s["num_clients"])),
        row("Records per client (min / median / max)",
            lambda n, s: f"{s['client_size_min']:,} / {s['client_size_median']:,} / {s['client_size_max']:,}"),
        row("Split per client", lambda n, s: "70 / 10 / 20"),
    ]
    for name, st in stats.items():
        print(f"{SOURCES[name][0]}: {st['n_total']} examples, {st['num_clients']} clients, sizes "
              f"{st['client_size_min']}/{st['client_size_median']}/{st['client_size_max']}, classes {st['class_counts_total']}")
    colspec = "p{2.3cm}" + "p{2.6cm}" * len(names)
    tex = (r"\begin{tabular}{" + colspec + "}\n\\toprule\n" + head + "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tab = PAPER_OUT / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "datasets.tex").write_text(tex, encoding="utf-8")
    print("wrote results/dataset_stats.json, paper/tables/datasets.tex, paper/figures/partition_*.pdf")


if __name__ == "__main__":
    main()
