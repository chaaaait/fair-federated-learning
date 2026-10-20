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
- Windows + GTX 1650: **to test in Block S2** (Claude Code). If native Windows fails: WSL2, else the recorded video.

## Talking points (for the examiners)
1. "These are four real processes talking over the network stack, not a loop in one program."
2. "Only the body travels — 140,064 parameters; the 1,290-parameter head never leaves the client."
3. "ε spent climbs each round and stops at the budget we set — that is the accountant."
4. "Jain's index at the end tells us how evenly the clients are served."
