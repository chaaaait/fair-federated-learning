"""ClientApp: one federated client = one Flower SuperNode.

Each round: load the shared body sent by the server, restore this client's private head from the node's
persistent state, run DP-SGD steps (per-example clipping + Gaussian noise, from `pfl.dp`), report the
privacy spent so far, and send back ONLY the body. The head never leaves the client (FedPer).
"""
from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pfl.data import build_federated_data
from pfl.dp import GradEngine, calibrate_sigma, epsilon_from
from pfl.metrics import accuracy
from pfl.utils import resolve_device

from .common import demo_config, fresh_model, split_state

app = ClientApp()
_CACHE: dict = {}


def _client_data(cfg: dict, pid: int):
    key = (cfg["dataset"], cfg["num_clients"], cfg["max_examples"], cfg["data_dir"], cfg["device"])
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = build_federated_data(cfg, resolve_device(cfg["device"]))
    data = _CACHE[key]
    return data, data.clients[pid]


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = demo_config(context.run_config)
    pid = int(context.node_config["partition-id"])
    data, client = _client_data(cfg, pid)
    device = client.y_tr.device
    rnd = int(msg.content["config"]["round"])

    model = fresh_model(cfg).to(device)
    state = model.state_dict()
    for name, t in msg.content["arrays"].to_torch_state_dict().items():
        state[name] = t.to(device)
    if "head" in context.state:  # this client's private head from earlier rounds
        for name, t in context.state["head"].to_torch_state_dict().items():
            state[name] = t.to(device)
    model.load_state_dict(state)

    steps_total = cfg["rounds"] * cfg["local_steps"]
    sigma = calibrate_sigma(cfg["epsilon"], cfg["sample_rate"], steps_total, 0, delta=cfg["delta"])
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] * 100_000 + pid * 1_000 + rnd)
    engine = GradEngine(model, cfg["grad_backend"], cfg["max_physical_batch"])
    for _ in range(cfg["local_steps"]):
        g = engine.private_grad(client.x_tr, client.y_tr, data.prep, cfg["sample_rate"], sigma, cfg["clip_norm"], gen)
        engine.add_flat_(g, -cfg["lr_local"])
    eps_spent = epsilon_from(sigma, cfg["sample_rate"], rnd * cfg["local_steps"], 0, delta=cfg["delta"])

    acc = accuracy(model, client.x_te, client.y_te, data.prep)
    body, head = split_state(model)
    context.state["head"] = ArrayRecord(head)  # stays on this node
    arrays = ArrayRecord(body)
    metrics = MetricRecord({
        "num-examples": client.n_tr,
        "acc": float(acc),
        "eps": float(eps_spent),
        "sigma": float(sigma),
        "bytes-up": int(arrays.count_bytes()),
        "bytes-down": int(msg.content["arrays"].count_bytes()),
        "partition": pid,
    })
    return Message(RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)
