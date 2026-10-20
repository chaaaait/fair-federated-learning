# Personalisation under Differential Privacy: which mechanism buys the most fairness per unit of privacy?

Code for *"A Controlled Comparison of Personalisation Mechanisms for Fairness in Differentially Private
Federated Learning"* — Chaitanya Patil, Aprark, Tanishq Jain, Dr. Kalaavathi B (VIT Vellore). Team T16, Project 20.

**Question.** Differential privacy (DP) hurts some federated clients much more than others. Personalisation softens
that cost, but the personalisation families have never been compared at one identical privacy budget. We compare
**fine-tuning, Per-FedAvg (meta-learning), IFCA-style clustering and FedPer (parameter decoupling)** — plus local-only
training, DP-FedAvg and the q-FFL fairness objective — with every client answering the same private queries, so every
method ends at the **same (ε, δ)**.

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
