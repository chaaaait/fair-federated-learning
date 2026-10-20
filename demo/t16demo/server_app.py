"""ServerApp: waits for the SuperNodes, runs DP FedPer rounds, prints one line per round, then a per-client table.

FedPer: the server averages only the shared body (weighted by client training-set size); heads stay on the clients.
"""
from __future__ import annotations

import time
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from .common import demo_config, fresh_model, split_state

app = ServerApp()


def say(text: str) -> None:
    log(INFO, text)


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = demo_config(context.run_config)
    k_expected, rounds = cfg["num_clients"], cfg["rounds"]
    body, head = split_state(fresh_model(cfg))
    params_body = sum(t.numel() for t in body.values())
    params_head = sum(t.numel() for t in head.values())

    t0 = time.time()
    nodes = list(grid.get_node_ids())
    while len(nodes) < k_expected and time.time() - t0 < 120:
        time.sleep(1)
        nodes = list(grid.get_node_ids())
    say(f"T16/P20 live demo | {len(nodes)} clients connected | DP FedPer | target eps={cfg['epsilon']:g}, "
        f"delta={cfg['delta']:g} | body {params_body:,} params shared, head {params_head:,} params kept on clients")
    say(f"{'round':>5} | {'mean acc':>8} | {'worst acc':>9} | {'eps spent':>9} | {'up MB':>6} | {'down MB':>7} | {'s':>5}")

    last = []
    for rnd in range(1, rounds + 1):
        t_round = time.time()
        msgs = [Message(RecordDict({"arrays": ArrayRecord(body), "config": ConfigRecord({"round": rnd})}),
                        dst_node_id=node, message_type=MessageType.TRAIN, group_id=str(rnd)) for node in nodes]
        replies = [m for m in grid.send_and_receive(msgs) if m.has_content()]
        if not replies:
            say("no replies received - check the SuperNode logs in logs/")
            return
        total = sum(int(m.content["metrics"]["num-examples"]) for m in replies)
        new_body = None
        for m in replies:
            w = int(m.content["metrics"]["num-examples"]) / total
            sd = m.content["arrays"].to_torch_state_dict()
            new_body = {k: v * w for k, v in sd.items()} if new_body is None else \
                {k: new_body[k] + v * w for k, v in sd.items()}
        body = new_body
        metrics = [m.content["metrics"] for m in replies]
        accs = [float(x["acc"]) for x in metrics]
        up = sum(int(x["bytes-up"]) for x in metrics) / 1e6
        down = sum(int(x["bytes-down"]) for x in metrics) / 1e6
        eps = max(float(x["eps"]) for x in metrics)
        say(f"{rnd:>5} | {100 * sum(accs) / len(accs):>7.1f}% | {100 * min(accs):>8.1f}% | {eps:>9.2f} | "
            f"{up:>6.2f} | {down:>7.2f} | {time.time() - t_round:>5.1f}")
        last = metrics

    accs = [float(x["acc"]) for x in last]
    jain = sum(accs) ** 2 / (len(accs) * sum(a * a for a in accs)) if any(accs) else float("nan")
    say("per-client accuracy after the last round (personalised model = shared body + own head):")
    for x in sorted(last, key=lambda x: int(x["partition"])):
        say(f"  client {int(x['partition'])}: {100 * float(x['acc']):.1f}%  (sigma = {float(x['sigma']):.2f})")
    say(f"Jain's fairness index = {jain:.3f} (1 = all clients equal) | total time {time.time() - t0:.0f}s")
