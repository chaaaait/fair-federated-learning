# Live demo — DP FedPer on a real Flower deployment (2–3 min)

**What it shows.** Four separate processes on one laptop: a Flower **SuperLink** (the server) and three Flower
**SuperNodes** (the clients), exchanging messages over HTTP on localhost. Each client trains with per-example
DP-SGD (clipping + Gaussian noise from `pfl.dp`), keeps its classifier head private (FedPer) and sends only the
shared body. The server prints one line per round:

```
round | mean acc | worst acc | eps spent | up MB | down MB |     s
    1 |    ..    |    ..     |   ..      |  1.69 |   1.69  |  ..
```
and finishes with each client's accuracy and **Jain's fairness index**. "eps spent" grows every round until it
reaches the target budget — the privacy accountant working live.

## Run it (PowerShell, repo root)
```powershell
.venv\Scripts\python -m pip install -e .                  # once: makes pfl importable for Flower's workers
.venv\Scripts\python demo\run_demo.py --dataset synthetic --device cpu --rounds 3   # self-test, no data
.venv\Scripts\python demo\run_demo.py                     # the real demo: CIFAR-10 subset, 5 rounds, eps = 4
.venv\Scripts\python demo\run_demo.py --device cpu        # use this while the GPU is busy with the sweep
```
Needs CIFAR-10 already in `data\` (run `scripts\dataset_stats.py` once) → works with Wi-Fi off.
Process logs: `logs\demo_superlink.log`, `logs\demo_supernode*.log`. Flower state: `logs\flwr_home\`.

## Tested
- Linux, Flower 1.40.0, synthetic data, CPU: 3 rounds, all three clients reply, ε reaches the target exactly. ✅
- Windows 11, Flower 1.41.0, CPU (21 Oct 2026, GPU busy with the sweep), no WSL needed: ✅
  - synthetic, 3 rounds: ε 2.50 → 3.33 → 4.00, Jain 0.855, 61 s total.
  - CIFAR-10 subset, 5 rounds: ε 3.00 → 4.00, mean acc 33.1%, worst 24.1%, Jain 0.964, **149 s total** (~29 s/round).
  - All processes stop cleanly; ports 8000/9094–9096 are freed. No firewall prompt appeared (localhost only).
- Windows 11, GTX 1650 (22 Oct 2026, after the sweep): CIFAR-10 subset, 5 rounds, ε → 4.00, Jain 0.968,
  **103 s total** (~20 s/round). Use the GPU in the review; `--device cpu` (149 s) is the fallback.

## Talking points (for the examiners)
1. "These are four real processes talking over the network stack, not a loop in one program."
2. "Only the body travels — 140,064 parameters; the 1,290-parameter head never leaves the client."
3. "ε spent climbs each round and stops at the budget we set — that is the accountant."
4. "Jain's index at the end tells us how evenly the clients are served."
