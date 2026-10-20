# CLAUDE.md — T16 / P20 (read this first, every session)

## What this is
VIT Computer Networks project, Team T16 / Project 20: **"A Controlled Comparison of Personalisation Mechanisms
for Fairness in Differentially Private Federated Learning."** Final Review **Wed 21 Oct 2026** (faculty cabin,
10–15 min + live demo). **Sat–Mon were lost → everything now runs on Tue 20 Oct (see RE-PLAN below).**

- Owner: **Chaitanya Patil** (laptop: Windows, PowerShell, **NVIDIA GTX 1650 4 GB**, driver 581.42; git installed; GitHub `chaaaait`).
- Plan: `docs/PLAN.md` (schedule, gates, queue, rubric map). Tracker: `docs/PROGRESS.md` — **update it after every block**.
  Chaitanya copies it into the claude.ai Project (`claude/PROGRESS.md`) when he switches back to a chat.
- Paper: `paper/` (IEEEtran journal template, LaTeX, compiled on Overleaf). References: `paper/refs.bib`, notes in `docs/REFERENCES.md`.

## ⚠️ RE-PLAN — Tue 20 Oct 2026 (overrides every date below)
Sat–Mon were lost (nothing ran yet). Review is **Wed 21 Oct**. Everything happens **today, Tue 20 Oct**.

| Time (Tue) | Block | Who |
|---|---|---|
| 08:00 | **S1** setup + tests + dataset stats + **Gate 1** (target ≤ 1.5 h) | Claude Code |
| ~09:30 | **Launch the sweep** in Chaitanya's own PowerShell window | Chaitanya |
| 09:30–19:00 | **S2** demo on CPU · **P1** Overleaf upload + check · env.txt / lock file | Claude Code |
| ~12:00 | **Gate 2** (ε₀; default 2, silence = 2) — decide BEFORE Q2 ends | Chaitanya |
| **19:00 HARD STOP** | Stop the sweep (Ctrl+C in its window) wherever it is → S3 | Chaitanya |
| 19:00–19:45 | **S3** aggregate → Gate 3 (`results/sanity.txt`) → figures | Claude Code |
| 19:45–22:00 | **P2** Results, Discussion, Conclusion, abstract sentence | Claude Code |
| 22:00–23:00 | **S4** GitHub push + v1.0 · deck in a claude.ai chat (figures + numbers) | both |
| 23:00 | PDF to Dr Kalaavathi B · Turnitin + own-voice pass | Chaitanya |
| Wed AM | 2 demo dry runs (Wi-Fi off) · backup video · rehearsal | team |

**Fit the sweep into 09:30–19:00 (570 min).** Gate 1 (`smoke.py`) prints the estimated minutes per CIFAR-10 run = *t*.
- Core = Q1–Q4 = **57 CIFAR runs** (never cut). Q5 = 30 Adult runs (small MLP, short). Q6 = 48 runs. Q7 = 12 runs.
- If 57·*t* > 300 min (*t* > 5.3) → set `rounds: 30` in `configs/base.yaml` **before launch** (never `local_steps`/`sample_rate`) and re-run smoke.
- Then cut until the total fits, in this order (pre-approved — just tell Chaitanya what was cut):
  1. drop Q7 · 2. Q6 `seeds: [0, 1]` (per-block `seeds:` key works) · 3. Q6 `eps: [8, 1]` · 4. Q5 `seeds: [0, 1]`.
- The sweep runs blocks in order (Q1→Q7), so a 19:00 stop always leaves the most important runs done.
- If the lock file survives Ctrl+C: next start with `--force` (or delete `results\.sweep.lock`).
- If a figure/table script fails because a block was cut, make it skip the missing data — never fill in numbers.

## How to work with Chaitanya
- Plain language, short sentences, bottom line first, **bold** key facts. Flag problems bluntly. No heavy maths.
- He must be able to defend every line in the paper and deck — explain choices conceptually when you make them.
- Ask before: changing the design (docs/PLAN.md §3), editing queue blocks other than `eps0` (exception: the
  RE-PLAN cut order is pre-approved), deleting results, pushing to GitHub for the first time.
- If the Windows username is not `Chaitanya`, use the real path wherever this file says `C:\Users\Chaitanya`.

## Hard rules
1. **Never invent or estimate a number.** Every number in the paper/deck comes from `results/*.csv` or `results/runs/*.json`.
2. **One heavy GPU job at a time.** While the sweep runs, test anything else (demo, scripts) with `device=cpu`.
3. **Long jobs run in Chaitanya's own PowerShell window**, not inside this session (they must survive session restarts).
   You monitor with `scripts/status.py` and `logs/sweep.log`.
4. Keep the code as is unless a test or run fails. If you change `pfl/`, re-run `tests/test_cpu.py`.
5. Commit after each block (`git add -A; git commit -m "..."`).

## Repo map
```
pfl/            library: config, data (CIFAR-10/Adult + Dirichlet), models (GroupNorm CNN, MLP), dp (Opacus
                per-example grads, clip+noise, RDP accountant, sigma calibration), methods (FedAvg+fine-tune,
                Local, Per-FedAvg, IFCA, FedPer, q-FFL), metrics (fairness, MIA, group gaps), engine (one run)
configs/        base.yaml (hyperparameters — change them ONLY here), queue.yaml (Q1–Q7, eps0 = Gate 2)
scripts/        run.py (one run) · sweep.py (queue, resumable) · status.py · smoke.py (Gate 1) ·
                dataset_stats.py (Table T2 + partition plot) · aggregate.py (CSV + Gate 3) · figures.py ·
                architecture.py (Fig. 1 system diagram — already rendered in paper/figures/)
tests/          test_cpu.py (synthetic data, ~1 min) · queue_test.yaml (tiny end-to-end queue)
demo/           Flower live demo (built + tested on Linux; Block S2 = test on Windows)
paper/          main.tex, sections/, refs.bib, figures/, tables/
docs/           PLAN.md, PROGRESS.md, REFERENCES.md
results/runs/   one JSON per finished run (commit these) · logs/ (not committed) · data/ (not committed)
```

## Blocks (in order)

### S1 — Setup + Gate 1 + launch (Tue 08:00)
1. Python: `py --version`; prefer 3.12 (3.10–3.13 fine). `py -3.12 -m venv .venv`
2. PyTorch with CUDA: `.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   Check: `.venv\Scripts\python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.cuda.get_arch_list())"`
   → must print `True` and include `sm_75`. If not, reinstall from `.../whl/cu126` (or the pytorch.org selector, a CUDA 12.x build).
3. `.venv\Scripts\python -m pip install -r requirements.txt` then `.venv\Scripts\python -m pip install -e .`
   (the second one makes `pfl` importable inside Flower's worker processes)
4. `.venv\Scripts\python tests\test_cpu.py` → all PASS.
5. `.venv\Scripts\python scripts\dataset_stats.py` → downloads CIFAR-10 (~170 MB) + UCI Adult; writes `paper/tables/datasets.tex`,
   `paper/figures/partition_*.pdf`. Paste the printed summary into PROGRESS.
6. `.venv\Scripts\python scripts\smoke.py` → **Gate 1**. If `time_budget` fails, lower `rounds` in `configs/base.yaml`
   (40 → 30), never `local_steps`/`sample_rate`; tell Chaitanya. If `backends_agree` fails, set `grad_backend: functorch`.
   Also run `scripts\smoke.py --dataset adult --rounds 2`.
7. Fill the `\todo{}` numbers in `paper/sections/04_setup.tex` (dataset sizes, clients, timing) from steps 5–6.
8. `git init`, `.gitignore` is ready, first commit.
9. Ask Chaitanya to open **his own PowerShell window** and run:
   `cd C:\Users\Chaitanya\T16_P20; .venv\Scripts\python scripts\sweep.py`
   Check `scripts\status.py` after the first run finishes (expect ~3–6 min per run). Laptop must stay plugged in and awake.

### S2 — Flower demo + reproducibility files (Tue, while the sweep runs; CPU only)
- The demo is **already built and tested on Linux** (Flower 1.40, synthetic data): `demo/` (`run_demo.py`,
  `t16demo/server_app.py`, `t16demo/client_app.py`, `pyproject.toml`). Read `demo/README.md`.
- Test on Windows **on CPU** while the GPU sweeps: `.venv\Scripts\python demo\run_demo.py --dataset synthetic --device cpu --rounds 3`,
  then `.venv\Scripts\python demo\run_demo.py --device cpu` (CIFAR-10 subset). Logs: `logs\demo_*.log`.
- Flower 1.40 facts learned while building it: SuperLink serves Fleet + Control APIs over **HTTP on one port**
  (`--port 8000`); SuperNodes use `--superlink 127.0.0.1:8000 --port 909x --node-config "partition-id=i num-partitions=3"`;
  `flwr run <app-dir> <connection>` reads connections from `$FLWR_HOME/config.toml` (run_demo.py writes it);
  Flower starts `flower-superexec` **by name**, so the venv's Scripts folder must be on PATH (run_demo.py does this).
- Likely Windows snags: process-tree cleanup (run_demo.py uses `taskkill /T`), ports already in use (change 8000/9094+),
  firewall prompt on first run (allow private networks). If it still fails after ~1 h → WSL2, else keep the video fallback.
- Time the real demo on GPU after the sweep (Tue evening); target ≤ 3 min for 5 rounds (lower `max-examples` in
  `demo/pyproject.toml` if slower).
- `env.txt` (nvidia-smi, python, torch/opacus/flwr versions) + `pip freeze > requirements-lock.txt`. Update README.

### Gate 2 (after Q1 + Q2 finish, ~3 h after launch)
`scripts\aggregate.py` then `scripts\figures.py` → open `paper/figures/m1_accuracy_vs_eps.png`. Pick ε₀ where DP-FedAvg
still learns but loses ≥ 5 points vs ∞ (default 2). Confirm with Chaitanya (silence = 2); edit `eps0` in `configs/queue.yaml`
(the sweep re-reads it before every run).

### P1 — Paper check (Tue afternoon)
Zip `paper/` for Overleaf (`paper_overleaf.zip`; upload via Overleaf → New Project → Upload Project). Apply Chaitanya's marks.

### S3 — Results (Tue 19:00, at the hard stop)
`status.py` → `sweep.py --retry-failed` if needed → `aggregate.py` → read `results/sanity.txt` (**Gate 3**) → `figures.py`.

### P2 — Paper results (Tue 19:45–22:00)
Write `paper/sections/05_results.tex`, `06_discussion.tex`, `07_conclusion.tex` and the abstract's results sentence,
following the TODO notes inside those files. Use only numbers from `results/summary_table.csv`, `eps_slopes.csv`,
`recommended.csv`. Include failure cases (clients hurt by personalisation, IFCA cluster collapse if any, low-ε collapse)
and limitations. Keep it ≤ 10 pages. Re-zip for Overleaf.

### S4 — GitHub (Tue ~22:00)
Chaitanya creates an empty public repo `t16-pfl-dp-fairness` at github.com/new (owner `chaaaait`, no README).
Then: `git remote add origin https://github.com/chaaaait/t16-pfl-dp-fairness.git; git branch -M main; git push -u origin main`
(he signs in when Git Credential Manager pops up). Finalise README (results summary, one command per figure), tag `v1.0`.

## Cut order if behind
Q7 → Q6 seeds 3→2 → Q6 ε 4→2 → Q5 Adult (state as a limitation). Never cut Q1–Q4, MIA, cost metrics, 3 seeds on core runs.
