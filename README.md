# Personalisation under Differential Privacy: which mechanism buys the most fairness per unit of privacy?

Code for *"A Controlled Comparison of Personalisation Mechanisms for Fairness in Differentially Private
Federated Learning"* — Chaitanya Patil, Aprark, Tanishq Jain, Dr. Kalaavathi B (VIT Vellore). Team T16, Project 20.

**Question.** Differential privacy (DP) hurts some federated clients much more than others. Personalisation softens
that cost, but the personalisation families have never been compared at one identical privacy budget. We compare
**fine-tuning, Per-FedAvg (meta-learning), IFCA-style clustering and FedPer (parameter decoupling)** — plus local-only
training, DP-FedAvg and the q-FFL fairness objective — with every client answering the same private queries, so every
method ends at the **same (ε, δ)**.

## Results in brief
CIFAR-10, 20 clients (Dirichlet α = 0.3), per-example DP-SGD. 392 runs, 0 failed, 251 private runs all within budget,
47.2 GPU hours (GTX 1650). At ε = 2 (mean ± SD over five seeds):

| Mechanism | Mean acc. (%) | Worst-10% acc. (%) | Jain | Gain vs DP-FedAvg (pts) | Total MB |
|---|---|---|---|---|---|
| DP-FedAvg (global) | 27.8 ± 1.6 | 12.5 ± 4.5 | 0.875 | — | 905 |
| Fine-tuning | 51.9 ± 0.6 | 36.2 ± 1.5 | 0.951 | +24.0 | 814 |
| Local only | 56.0 ± 0.5 | 37.9 ± 1.3 | 0.960 | +28.2 | 0 |
| Per-FedAvg | 49.7 ± 1.1 | 32.1 ± 3.1 | 0.939 | +21.9 | 905 |
| Clustered (IFCA) | 43.5 ± 2.2 | 26.3 ± 5.0 | 0.945 | +15.7 | 1809 |
| FedPer | 53.6 ± 0.9 | 37.4 ± 1.4 | 0.956 | +25.7 | 896 |
| MR-MTL (λ on validation) | 55.7 ± 0.5 | 37.9 ± 1.3 | 0.959 | +27.9 | 905 |

- **Cost of DP, in steps** (FedAvg, seeds 0–2): 200 → 40 non-private rounds costs 19.5 pts of mean accuracy, clipping
  to C = 1 without noise a further 10.9, the noise at ε = 2 only 2.1. With the same 200 rounds, ε = 2 still costs 26.2 pts
  (mean) and 28.1 pts (worst-10%). Clipping costs local-only training 2.2 pts and FedPer 7.2.
- **DP removes the benefit of federation under strong label skew:** without DP FedPer beats local-only training (76.4% vs
  71.3%, 200 rounds); under DP local-only training is ahead at every budget (paired difference at ε = 2: +2.4 pts, 95% CI
  1.6–3.2, all five seeds), also with 80/200 private rounds, C = 8 and the learning rate picked on validation. MR-MTL
  (Liu et al. 2022) picks almost no federation (λ = 0.01). With milder skew (α = 1.0) FedPer keeps a small lead (40.4% vs 39.1%).
- Below ε = 1, fine-tuning and Per-FedAvg lose the least accuracy per unit ε and keep the worst clients highest.
- q-FFL does not lift the weakest clients at any constant L (best: L = 0.01/η, 22.8% mean at ε = 2, worst-10% 1.5%).
- Membership-inference AUC stays at 0.499–0.509 for all private runs; without DP (200 rounds) it reaches 0.660 for local-only.

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
.venv\Scripts\python scripts\figures.py         # paper\figures\*.pdf|png, results\tables\*.tex
```
One run: `.venv\Scripts\python scripts\run.py --method fedper --eps 2 --seed 0`

Every figure and table comes from the committed run files in `results\runs\`, so you can rebuild them without
re-running the sweep:

| Paper item | File(s) | Script | Data |
|---|---|---|---|
| Fig. 1 partition (Appendix A) | `paper/figures/partition_*` | `scripts\dataset_stats.py` | `results/dataset_stats.json` |
| Fig. 2 system diagram | `paper/figures/architecture.*` | `scripts\architecture.py` | — |
| Fig. 3 non-private convergence | `nonprivate_convergence` | `scripts\figures.py` | `results/curves_nonprivate.csv` |
| Fig. 4 DP-FedAvg / fine-tuning vs ε | `m1_accuracy_vs_eps` | `scripts\figures.py` | `results/summary_table.csv`, `summary_converged.csv` |
| Fig. 5 cost of DP in steps | `cost_of_dp_steps` | `scripts\figures.py` | `results/cost_decomposition.csv` |
| Figs. 6–7 mechanisms at ε₀, privacy sweep | `m2_mechanisms_at_eps0`, `m2_accuracy_vs_eps_all` | `scripts\figures.py` | `results/summary_table.csv` |
| Fig. 8 q-FFL vs constant L | `m3_qffl_L_sweep` | `scripts\figures.py` | `results/summary_table.csv` |
| Fig. 9 membership inference (Appendix A) | `privacy_mia_vs_eps` | `scripts\figures.py` | `results/summary_table.csv` |
| Fig. 10 MR-MTL vs λ (Appendix A) | `mrmtl_lambda` | `scripts\figures.py` | `results/mrmtl_lambda.csv` |
| Table 4 200-round reference | — | `scripts\aggregate.py` | `results/summary_converged.csv` |
| Table 5 main results, paired CIs | `results/tables/main_results.tex` | `scripts\aggregate.py` | `results/summary_table.csv`, `paired_eps0.csv` |
| Table 6 Adult | `results/tables/adult_group_fairness.tex` | `scripts\aggregate.py` | `results/summary_table.csv`, `summary_converged.csv` |
| Table 7 robustness checks | — | `scripts\aggregate.py` | `results/summary_aux.csv`, `lr_check.csv` |
| Table 8 recommendations | `results/tables/recommended.tex` | `scripts\aggregate.py` | `results/recommended.csv` |
| §4.8 communication and compute cost | `results/tables/cost.tex` | `scripts\aggregate.py` | `results/summary_table.csv` |
| FedPer + q-FFL explanation (§4.5) | — | `scripts\diag_fedper_qffl.py` | `results/diag_fedper_qffl.csv` |

Figures are written to `paper/figures/` as PDF and PNG. Run counts, budget checks and GPU hours: `results/sanity.txt`.

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
`pfl/` library · `configs/` · `scripts/` · `tests/` · `demo/` · `paper/` (figures + `refs.bib`; the manuscript itself is a Word file, not in the repo) · `results/runs/` (one JSON per run)

## Use of generative AI
Claude (Anthropic) was used to help write and test the experiment code and figure scripts, run and aggregate
the experiment sweep, make the figures, and draft and edit parts of the paper. The authors designed the study,
checked every reported number against the run files in `results/`, verified the references, and take full
responsibility for the work.

## Licence
Code: MIT. CIFAR-10: free research use with citation (Krizhevsky, 2009). Adult: UCI ML Repository, CC BY 4.0.
