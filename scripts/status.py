"""Progress of the queue: done / failed / pending per block, average run time, ETA, last log lines.

  .venv\\Scripts\\python scripts\\status.py
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from _common import LOGS, QUEUE_YAML, RESULTS, expand_queue
from pfl.utils import read_json


def main():
    specs = expand_queue(QUEUE_YAML)
    per = defaultdict(lambda: {"done": 0, "failed": 0, "pending": 0})
    times = []
    for s in specs:
        if s.result_path.exists():
            per[s.block]["done"] += 1
            try:
                times.append(read_json(s.result_path)["cost"]["time_total_s"])
            except Exception:
                pass
        elif s.error_path.exists():
            per[s.block]["failed"] += 1
        else:
            per[s.block]["pending"] += 1
    print(f"{'block':<6}{'done':>6}{'failed':>8}{'pending':>9}")
    for block in sorted(per):
        p = per[block]
        print(f"{block:<6}{p['done']:>6}{p['failed']:>8}{p['pending']:>9}")
    total_done = sum(p["done"] for p in per.values())
    total_pending = sum(p["pending"] for p in per.values())
    total_failed = sum(p["failed"] for p in per.values())
    print(f"{'all':<6}{total_done:>6}{total_failed:>8}{total_pending:>9}   ({len(specs)} runs in queue)")
    n_wait = sum(s.waiting for s in specs)
    if n_wait:
        print(f"({n_wait} of the pending MR-MTL runs wait for their lambda grid to finish)")
    if times:
        avg = sum(times) / len(times)
        eta = dt.datetime.now() + dt.timedelta(seconds=avg * total_pending)
        print(f"avg run {avg/60:.1f} min -> remaining ~{avg*total_pending/3600:.1f} h (ETA {eta:%a %H:%M})")
    lock = RESULTS / ".sweep.lock"
    print("sweep running: " + ("yes" if lock.exists() else "no"))
    log = LOGS / "sweep.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8").splitlines()[-6:]
        print("\nlast log lines:\n  " + "\n  ".join(lines))


if __name__ == "__main__":
    main()
