# Personalisation under Differential Privacy: which mechanism buys the most fairness per unit of privacy?

Code for *"A Controlled Comparison of Personalisation Mechanisms for Fairness in Differentially Private
Federated Learning"* — Chaitanya Patil, Aprark, Tanishq Jain, Dr. Kalaavathi B (VIT Vellore). Team T16, Project 20.

**Question.** Differential privacy (DP) hurts some federated clients much more than others. Personalisation softens
that cost, but the personalisation families have never been compared at one identical privacy budget. We compare
**fine-tuning, Per-FedAvg (meta-learning), IFCA-style clustering and FedPer (parameter decoupling)** — plus local-only
training, DP-FedAvg and the q-FFL fairness objective — with every client answering the same private queries, so every
method ends at the **same (ε, δ)**.

## Results in brief
CIFAR-10, 20 clients (Dirichlet α = 0.3), per-example DP-SGD, mean over 3 seeds (207 runs, 0 failed: 147 main runs
plus 60 runs for longer non-private training and the q-FFL constant). At ε = 2:

| Mechanism | Mean acc. (%) | Worst-10% acc. (%) | Jain | Gain vs DP-FedAvg (pts) | Total MB |
|---|---|---|---|---|---|
| DP-FedAvg (global) | 27.2 ± 1.7 | 12.1 ± 6.2 | 0.865 | — | 905 |
| Fine-tuning | 51.8 ± 0.9 | 36.2 ± 2.0 | 0.950 | +24.6 | 814 |
| Local only | 56.2 ± 0.6 | 38.1 ± 1.4 | 0.960 | +29.0 | 0 |
| Per-FedAvg | 49.9 ± 0.8 | 32.8 ± 2.1 | 0.942 | +22.7 | 905 |
| Clustered (IFCA) | 42.8 ± 2.7 | 24.0 ± 5.3 | 0.940 | +15.6 | 1809 |
| FedPer | 54.1 ± 0.4 | 38.1 ± 0.8 | 0.959 | +26.8 | 896 |

- DP hits the weakest clients hardest: against non-private FedAvg trained for 200 rounds (59.8% mean, 47.3% worst-10%),
  ε = 2 costs at least 32.5 pts of mean and 35.2 pts of worst-10% accuracy (54% vs 74% of their value).
- Without DP, federation helps (200 rounds: FedPer 76.4% vs local-only 71.3%); under DP it does not: local-only training
  has the highest mean at every private budget. FedPer is the best federated option and improves every client in every seed.
- Below ε = 1, fine-tuning and Per-FedAvg lose the least accuracy per unit ε and keep the worst clients highest.
- q-FFL does not lift the weakest clients: with the default constant L = 1/η it stalls under DP (12.5% mean at ε = 2);
  a smaller L (0.01/η) recovers 23.4%, but its worst-10% accuracy stays below DP-FedAvg's at every L tested.
- Membership-inference AUC stays at 0.500–0.509 for all private runs; without DP (200 rounds) it reaches 0.660 for local-only.

Source: `results/summary_table.csv` (all metrics, main protocol), `results/summary_converged.csv` (200-round non-private
runs), `results/curves_nonprivate.csv`, `results/recommended.csv`, `results/eps_slopes.csv`; details in the paper.

## What is measured
| Group | Metrics |
|---|---|
| Utility | mean client accuracy; personalisation gain over the DP global model |
| Fairness | worst-10% client accuracy, SD and variance across clients, Jain's index; demographic-parity and equal-opportunity gaps (Adult, by sex) |
| Privacy | ε and δ (Opacus RDP accountant); loss-threshold membership-inference AUC / accuracy / advantage |
| Cost | MB per round and total, seconds per round, peak GPU memory, DP time overhead |

## Setup (Windows, PowerShell; tested on a GTX 1650 4 GB)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .            # pfl importable everywhere (needed by the Flower demo)
.venv\Scripts\python tests\test_cpu.py          # ~1 min, synthetic data, no download
```
Exact versions: `requirements-lock.txt` and `env.txt`. Linux/macOS: use `.venv/bin/python` and `/` paths.

## Reproduce
```powershell
.venv\Scripts\python scripts\dataset_stats.py   # downloads CIFAR-10 + UCI Adult; dataset table + partition plot
.venv\Scripts\python scripts\smoke.py           # Gate 1: GPU, gradients, every method, accountant, timing
.venv\Scripts\python scripts\sweep.py           # all experiments in configs\queue.yaml (resumable)
.venv\Scripts\python scripts\status.py          # progress + ETA
.venv\Scripts\python scripts\aggregate.py       # results\summary_*.csv, eps_slopes.csv, recommended.csv, sanity.txt
.venv\Scripts\python scripts\figures.py         # paper\figures\*.pdf|png, paper\tables\*.tex
```
One run: `.venv\Scripts\python scripts\run.py --method fedper --eps 2 --seed 0`

Every figure and table comes from the committed run files in `results\runs\`, so you can rebuild them without
re-running the sweep:

| Paper item | File | Command |
|---|---|---|
| Fig. 1 system diagram | `paper/figures/architecture.*` | `scripts\architecture.py` |
| Dataset table, partition plots | `paper/tables/datasets.tex`, `paper/figures/partition_*` | `scripts\dataset_stats.py` |
| All result figures (M1–M3, MIA) and Tables main / Adult / cost / recommended | `paper/figures/m*`, `privacy_mia_vs_eps`, `paper/tables/*.tex` | `scripts\aggregate.py` then `scripts\figures.py` |

If the CIFAR-10 download fails with `CERTIFICATE_VERIFY_FAILED` (seen with Python's SSL store on Windows), fetch the
archive with the system `curl` instead; torchvision checks its md5 and extracts it on the next run:
`curl.exe -L -o data\cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

Hyperparameters live in one place: `configs/base.yaml`. The experiment list is `configs/queue.yaml`.

## Design in one paragraph
CIFAR-10 (60k images) is split over 20 clients with a label Dirichlet(α = 0.3) partition; UCI Adult is split by
occupation (one client per occupation with ≥ 500 rows). Each client splits its data 70/10/20 (train/val/test).
Every client trains with per-example DP-SGD (per-record gradients clipped to norm 1, Gaussian noise, Poisson sampling
rate 0.03). All clients take part in every round, and the noise multiplier is calibrated per run from the exact number
of private queries (gradient steps plus noised loss reports for q-FFL and IFCA), so every configuration spends the same
(ε, δ = 10⁻⁵). Fine-tuning branches from the FedAvg run so that both use the same number of steps. The model is a small
GroupNorm CNN (~0.14 M parameters) — BatchNorm would break per-example DP.

## Live demo
`demo/` — Flower deployment (1 SuperLink + 3 SuperNodes talking HTTP on localhost) running DP FedPer for 5 rounds.
See `demo/README.md`.

## Layout
`pfl/` library · `configs/` · `scripts/` · `tests/` · `demo/` · `paper/` (IEEE LaTeX) · `results/runs/` (one JSON per run)

## Licence
Code: MIT. CIFAR-10: free research use with citation (Krizhevsky, 2009). Adult: UCI ML Repository, CC BY 4.0.
