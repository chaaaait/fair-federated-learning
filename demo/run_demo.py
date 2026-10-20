"""Live demo: start a local Flower deployment, run the DP FedPer app, stop everything.

  .venv\\Scripts\\python demo\\run_demo.py                      # CIFAR-10 subset (data\\ must already hold CIFAR-10)
  .venv\\Scripts\\python demo\\run_demo.py --device cpu         # while the GPU is busy with the sweep
  .venv\\Scripts\\python demo\\run_demo.py --dataset synthetic  # self-test, no data needed

What runs: 1 flower-superlink + 3 flower-supernode processes on localhost (the clients), then `flwr run`
submits the app. Process logs go to logs\\demo_*.log; the round table is streamed to this terminal.
Requires: `pip install -e .` once (makes `pfl` importable in Flower's worker processes).
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
SUPERLINK_PORT = 8000  # Flower 1.40: Fleet + Control APIs on one HTTP port (flower-superlink --port)


def exe(name: str) -> str:
    for cand in (BIN / name, BIN / f"{name}.exe"):
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"'{name}' not found - install flwr into this venv first")


def start(cmd: list[str], log_name: str, env: dict) -> subprocess.Popen:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(cmd, stdout=open(logs / log_name, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            env=env, cwd=ROOT, **kwargs)


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":  # kill the whole tree (SuperLink/SuperNode spawn worker processes)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=4.0)
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "synthetic"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--clients", type=int, default=3)
    args = ap.parse_args()

    home = ROOT / "logs" / "flwr_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        '[superlink]\ndefault = "demo"\n\n'
        f'[superlink.demo]\naddress = "127.0.0.1:{SUPERLINK_PORT}"\ninsecure = true\n', encoding="utf-8")
    env = dict(os.environ, FLWR_HOME=str(home), PYTHONUNBUFFERED="1")
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")  # Flower starts helper processes by name

    procs = []
    try:
        print(f"starting SuperLink on 127.0.0.1:{SUPERLINK_PORT} ...", flush=True)
        procs.append(start([exe("flower-superlink"), "--insecure", "--port", str(SUPERLINK_PORT)],
                           "demo_superlink.log", env))
        time.sleep(4)
        for i in range(args.clients):
            print(f"starting SuperNode {i} (client {i}) ...", flush=True)
            procs.append(start([exe("flower-supernode"), "--insecure", "--superlink", f"127.0.0.1:{SUPERLINK_PORT}",
                                "--port", str(9094 + i), "--node-config",
                                f"partition-id={i} num-partitions={args.clients}"], f"demo_supernode{i}.log", env))
        time.sleep(4)
        run_config = (f'num-server-rounds={args.rounds} epsilon={args.epsilon} dataset="{args.dataset}" '
                      f'device="{args.device}" num-partitions={args.clients} '
                      f'data-dir="{(ROOT / "data").as_posix()}"')
        cmd = [exe("flwr"), "run", str(ROOT / "demo"), "demo", "--stream", "--run-config", run_config]
        print("submitting the app: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, cwd=ROOT, check=False)
    finally:
        for p in reversed(procs):
            stop(p)
        print("demo processes stopped.", flush=True)


if __name__ == "__main__":
    main()
