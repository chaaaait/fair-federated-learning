"""Run the experiment queue (configs/queue.yaml). Safe to stop and restart at any time.

  .venv\\Scripts\\python scripts\\sweep.py                 # everything, in queue order
  .venv\\Scripts\\python scripts\\sweep.py --only Q1 Q2    # selected blocks
  .venv\\Scripts\\python scripts\\sweep.py --dry-run       # list what would run
  .venv\\Scripts\\python scripts\\sweep.py --retry-failed  # re-attempt runs that crashed

- Finished runs (results/runs/<id>.json) are skipped; a crash writes <id>.error.json and the sweep moves on.
- The queue file is re-read before every run, so changing eps0 (Gate 2) affects runs not yet started.
- MR-MTL runs with `mrmtl_lambda: auto` wait until their lambda grid has finished, then use the value with the
  best validation accuracy (scripts/_common.py, resolve_auto_lambda).
- A lock file stops two sweeps from sharing the GPU. It counts as stale after 30 min without a heartbeat.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import time
import traceback

import torch

from _common import LOGS, QUEUE_YAML, RESULTS, RUNS, config_for, expand_queue
from pfl.engine import run_experiment
from pfl.utils import write_json_atomic

LOCK = RESULTS / ".sweep.lock"
STALE_SECONDS = 30 * 60


def log_line(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOGS / "sweep.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def acquire_lock(force: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < STALE_SECONDS and not force:
            raise SystemExit(f"Another sweep looks active (lock touched {age/60:.0f} min ago): {LOCK}\n"
                             "If it is not running, delete the lock file or pass --force.")
    LOCK.write_text(f"pid={os.getpid()} started={dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")


def heartbeat() -> None:
    os.utime(LOCK, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=str(QUEUE_YAML))
    ap.add_argument("--only", nargs="*", default=None, help="block names, e.g. Q1 Q2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore an existing lock file")
    args = ap.parse_args()

    if args.dry_run:
        specs = expand_queue(args.queue, args.only)
        todo = [s for s in specs if not s.result_path.exists()]
        for s in todo:
            tag = "FAILED-before" if s.error_path.exists() else ("WAITING (lambda picked after its grid)"
                                                                 if s.waiting else "")
            print(f"{s.block:<4} {s.run_id} {tag}")
        print(f"\n{len(todo)} to run, {len(specs) - len(todo)} done, {len(specs)} total")
        return

    acquire_lock(args.force)
    RUNS.mkdir(parents=True, exist_ok=True)
    attempted, n_run = set(), 0
    log_line(f"sweep start (pid {os.getpid()})")
    try:
        while True:
            specs = expand_queue(args.queue, args.only)
            pending = [s for s in specs if not s.result_path.exists() and s.run_id not in attempted
                       and not s.waiting and (args.retry_failed or not s.error_path.exists())]
            if not pending:
                stuck = [s.run_id for s in specs if s.waiting]
                if stuck:
                    log_line(f"{len(stuck)} auto runs never got their lambda (a grid run failed?): {stuck}")
                break
            spec = pending[0]
            attempted.add(spec.run_id)
            heartbeat()
            done = sum(s.result_path.exists() for s in specs)
            log_line(f"START {spec.block} {spec.run_id} ({done}/{len(specs)} done)")
            t0 = time.perf_counter()
            try:
                cfg = config_for(spec)
                res = run_experiment(cfg)
                res["run_id"] = spec.run_id
                res["blocks"] = cfg["blocks"]
                write_json_atomic(spec.result_path, res)
                if spec.error_path.exists():
                    spec.error_path.unlink()
                parts = []
                for name, v in res["views"].items():
                    parts.append(f"{name}={v['summary']['mean']:.3f}/{v['summary']['worst10']:.3f}")
                log_line(f"END   {spec.run_id} {time.perf_counter() - t0:.0f}s eps={res['privacy']['epsilon_spent']} "
                         + " ".join(parts))
            except Exception as exc:  # keep going; the error file records what happened
                write_json_atomic(spec.error_path, {"run_id": spec.run_id, "error": repr(exc),
                                                    "traceback": traceback.format_exc()})
                log_line(f"FAIL  {spec.run_id}: {exc!r}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            n_run += 1
            if args.max_runs and n_run >= args.max_runs:
                break
    finally:
        if LOCK.exists():
            LOCK.unlink()
        log_line(f"sweep stop after {n_run} runs")


if __name__ == "__main__":
    main()
