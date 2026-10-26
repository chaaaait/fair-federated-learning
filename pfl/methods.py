"""Federated algorithms. Every client takes part in every round (20 clients; no client sampling),
so every client answers exactly the same number of private queries -> identical epsilon.

Method        Shares with server        Personalised model per client
------------  ------------------------  -----------------------------------------------------------
fedavg        full model                none (global model)      + "finetune" view: branch at round
                                                                   R - r, then r*S local DP steps (D10)
local         nothing                   own model trained only on own data
perfedavg     full model                meta-model + a few DP adaptation steps (first-order Per-FedAvg)
ifca          full model of 1 cluster   the cluster model it picks by (noisy) lowest loss
fedper        body only                 shared body + head that never leaves the client
mrmtl         full model                own model, pulled towards the server mean by lam/2 ||w_k - w_bar||^2
                                        (MR-MTL, Liu et al. 2022, Alg. A1); lam = 0 is local training
q-FFL         aggregation="qffl" with fedavg or fedper: server upweights high-loss clients (q-FedAvg)

clip_only=True (control, eps = inf): per-record gradients are clipped and batches Poisson-sampled exactly as in
DP-SGD, but no noise is added, so a run isolates the cost of clipping from the cost of noise.
"""
from __future__ import annotations

import numpy as np
import torch

from .metrics import accuracy, fairness_summary, record_losses
from .models import build_model, get_flat, head_mask, set_flat
from .utils import Stopwatch

BYTES_PER_PARAM = 4  # float32 on the wire


# ----------------------------------------------------------------------------- context
class Ctx:
    """Everything a method needs: config, data, model, gradient engine, noise level, RNG, logs."""

    def __init__(self, cfg, data, model, engine, device, sigma: float, gen: torch.Generator, log=print):
        self.cfg, self.data, self.model, self.engine, self.device = cfg, data, model, engine, device
        self.sigma = float(sigma)
        self.private = self.sigma > 0
        self.clip_only = bool(cfg.get("clip_only", False)) and not self.private
        self.sigma_loss = cfg["loss_noise_ratio"] * self.sigma if self.private else 0.0
        self.gen = gen
        self.log = log
        self.R, self.S = cfg["rounds"], cfg["local_steps"]
        self.q, self.C, self.lr = cfg["sample_rate"], cfg["clip_norm"], cfg["lr_local"]
        self.clients = data.clients
        self.K = len(self.clients)
        self.P = engine.P
        self.hmask = head_mask(model).to(device)
        self.P_head = int(self.hmask.sum().item())
        self.round_times: list[float] = []
        self.curves = {"round": [], "val_mean": [], "val_worst10": []}
        self.qcount = {c.cid: {"grad": 0, "loss": 0} for c in self.clients}
        self.sw = Stopwatch(device)

    # -- private queries ------------------------------------------------------------------
    def grad_query(self, client):
        self.qcount[client.cid]["grad"] += 1
        if self.private or self.clip_only:  # clip_only: sigma = 0, so private_grad clips but adds no noise
            return self.engine.private_grad(client.x_tr, client.y_tr, self.data.prep, self.q,
                                            self.sigma, self.C, self.gen)
        return self.engine.plain_grad(client.x_tr, client.y_tr, self.data.prep, self.q, self.gen)

    def train(self, vec, client, steps: int, lr: float):
        """`steps` (private) SGD steps on the client's training split, starting from `vec`."""
        set_flat(self.model, vec)
        for _ in range(steps):
            g = self.grad_query(client)
            if g is not None:
                self.engine.add_flat_(g, -lr)
        return get_flat(self.model)

    def loss_query(self, vec, client) -> float:
        """Mean training loss of `vec` on the client. Private runs: per-record losses clipped to
        [0, loss_clip], summed, Gaussian noise (sigma_loss * loss_clip) added, divided by n.
        clip_only runs: clipped the same way, no noise."""
        self.qcount[client.cid]["loss"] += 1
        set_flat(self.model, vec)
        losses = record_losses(self.model, client.x_tr, client.y_tr, self.data.prep)
        if not (self.private or self.clip_only):
            return float(losses.mean())
        clip = self.cfg["loss_clip"]
        total = losses.clamp(0.0, clip).sum()
        if self.private:
            total = total + torch.randn((), generator=self.gen, device=self.device) * (self.sigma_loss * clip)
        return float(total / client.n_tr)

    # -- logging --------------------------------------------------------------------------
    def log_curve(self, r: int, vec_for) -> None:
        accs = []
        for i, c in enumerate(self.clients):
            set_flat(self.model, vec_for(i))
            accs.append(accuracy(self.model, c.x_va, c.y_va, self.data.prep))
        s = fairness_summary(accs)
        self.curves["round"].append(r)
        self.curves["val_mean"].append(s["mean"])
        self.curves["val_worst10"].append(s["worst10"])
        rt = np.mean(self.round_times[-5:]) if self.round_times else float("nan")
        self.log(f"  round {r:3d}/{self.R} | val mean {s['mean']:.3f} | worst10 {s['worst10']:.3f} | {rt:.2f} s/round")

    def should_log(self, r: int) -> bool:
        return (r + 1) % self.cfg["eval_every"] == 0 or r + 1 == self.R

    def comm(self, kind: str, rounds: int, k: int = 1, extra_up: int = 0) -> dict:
        P, Pb = self.P, self.P - self.P_head
        down, up = {"full": (P, P), "base": (Pb, Pb), "clusters": (k * P, P), "none": (0, 0)}[kind]
        up += extra_up
        per_round = self.K * (down + up) * BYTES_PER_PARAM
        return {
            "kind": kind, "rounds": rounds,
            "params_down_per_client": down, "params_up_per_client": up,
            "bytes_down_per_round": self.K * down * BYTES_PER_PARAM,
            "bytes_up_per_round": self.K * up * BYTES_PER_PARAM,
            "MB_per_round": per_round / 1e6,
            "total_MB": per_round * rounds / 1e6,
        }


# ----------------------------------------------------------------------------- aggregation
class MeanAgg:
    """FedAvg: data-size-weighted mean of client updates."""

    def __init__(self, lr_server: float):
        self.lr, self.num, self.den = lr_server, None, 0.0

    def add(self, delta: torch.Tensor, weight: float) -> None:
        self.num = delta * weight if self.num is None else self.num + delta * weight
        self.den += weight

    def result(self, w: torch.Tensor) -> torch.Tensor:
        return w.clone() if self.num is None else w + self.lr * self.num / self.den


class QFFLAgg:
    """q-FedAvg (Li et al., ICLR 2020): Delta w_k = L (w - w_k), weights F_k^q, step normalised by h_k."""

    def __init__(self, q: float, lipschitz: float):
        self.q, self.L, self.num, self.den = float(q), float(lipschitz), None, 0.0

    def add(self, delta: torch.Tensor, loss: float) -> None:
        f = max(float(loss), 1e-4)
        dw = -self.L * delta
        contrib = (f ** self.q) * dw
        self.num = contrib if self.num is None else self.num + contrib
        h = self.L * f ** self.q
        if self.q > 0:
            h += self.q * f ** (self.q - 1) * float(dw.pow(2).sum())
        self.den += h

    def result(self, w: torch.Tensor) -> torch.Tensor:
        return w.clone() if self.num is None else w - self.num / self.den


def _aggregator(ctx):
    if ctx.cfg["aggregation"] == "qffl":
        return QFFLAgg(ctx.cfg["qffl_q"], ctx.cfg.get("qffl_L_scale", 1.0) / ctx.lr)
    return MeanAgg(ctx.cfg["lr_server"])


def _qffl_suffix(cfg) -> str:
    """View-name suffix for a non-default q-FFL constant L, so tuned runs never collide with the default ones."""
    s = float(cfg.get("qffl_L_scale", 1.0))
    return "" if s == 1.0 else f"_L{s:g}"


# ----------------------------------------------------------------------------- methods
def run_fedavg(ctx: Ctx, w0: torch.Tensor) -> dict:
    cfg, R, S = ctx.cfg, ctx.R, ctx.S
    qffl = cfg["aggregation"] == "qffl"
    ft = bool(cfg.get("ft_branch", True)) and not qffl
    r_ft = max(1, int(round(cfg["ft_fraction"] * R))) if ft else 0
    w, w_branch = w0.clone(), None
    for r in range(R):
        if ft and r == R - r_ft:
            w_branch = w.clone()
        ctx.sw.start()
        agg = _aggregator(ctx)
        for c in ctx.clients:
            f_k = ctx.loss_query(w, c) if qffl else None
            v = ctx.train(w, c, S, ctx.lr)
            agg.add(v - w, f_k if qffl else c.n_tr)
        w = agg.result(w)
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, lambda i: w)
    name = f"qffl_q{cfg['qffl_q']:g}{_qffl_suffix(cfg)}" if qffl else "fedavg"
    views = {name: {"vecs": [w] * ctx.K, "comm": ctx.comm("full", R, extra_up=1 if qffl else 0),
                    "global": True}}
    if ft:
        vecs = [ctx.train(w_branch, c, r_ft * S, ctx.lr) for c in ctx.clients]
        views["finetune"] = {"vecs": vecs, "comm": ctx.comm("full", R - r_ft),
                             "note": f"branch after round {R - r_ft}; {r_ft * S} local steps"}
    return views


def run_local(ctx: Ctx, w0: torch.Tensor) -> dict:
    vecs = [w0.clone() for _ in ctx.clients]
    for r in range(ctx.R):
        ctx.sw.start()
        for i, c in enumerate(ctx.clients):
            vecs[i] = ctx.train(vecs[i], c, ctx.S, ctx.lr)
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, lambda i: vecs[i])
    return {"local": {"vecs": vecs, "comm": ctx.comm("none", 0)}}


def _perfedavg_local(ctx: Ctx, w, c, meta_steps: int, alpha: float, beta: float):
    """First-order Per-FedAvg: adapt on batch 1, take the meta-gradient on batch 2 at the adapted point."""
    set_flat(ctx.model, w)
    for _ in range(meta_steps):
        theta = get_flat(ctx.model)
        g1 = ctx.grad_query(c)
        if g1 is not None:
            ctx.engine.add_flat_(g1, -alpha)
        g2 = ctx.grad_query(c)
        set_flat(ctx.model, theta)
        if g2 is not None:
            ctx.engine.add_flat_(g2, -beta)
    return get_flat(ctx.model)


def _meta_steps(cfg) -> int:
    """Per-FedAvg meta-steps per round (2 gradient queries each). Default S/2 = the query count of FedAvg."""
    return int(cfg.get("perfedavg_meta_steps") or cfg["local_steps"] // 2)


def run_perfedavg(ctx: Ctx, w0: torch.Tensor) -> dict:
    cfg = ctx.cfg
    alpha, beta, adapt = cfg["perfedavg_alpha"], cfg["perfedavg_beta"], cfg["perfedavg_adapt_steps"]
    meta_steps = _meta_steps(cfg)
    w = w0.clone()
    for r in range(ctx.R):
        ctx.sw.start()
        agg = MeanAgg(cfg["lr_server"])
        for c in ctx.clients:
            v = _perfedavg_local(ctx, w, c, meta_steps, alpha, beta)
            agg.add(v - w, c.n_tr)
        w = agg.result(w)
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, lambda i: w)
    vecs = [ctx.train(w, c, adapt, alpha) for c in ctx.clients]
    return {"perfedavg": {"vecs": vecs, "comm": ctx.comm("full", ctx.R)}}


def run_ifca(ctx: Ctx, w0: torch.Tensor) -> dict:
    cfg, k = ctx.cfg, ctx.cfg["ifca_k"]
    models = [w0.clone()]
    for j in range(1, k):  # distinct initialisations for the other cluster models
        torch.manual_seed(cfg["seed"] * 1000 + j)
        models.append(get_flat(build_model(cfg, ctx.data).to(ctx.device)))
    assign = np.zeros(ctx.K, dtype=int)
    hist = []
    for r in range(ctx.R):
        ctx.sw.start()
        aggs = [MeanAgg(cfg["lr_server"]) for _ in range(k)]
        for i, c in enumerate(ctx.clients):
            losses = [ctx.loss_query(models[j], c) for j in range(k)]
            j = int(np.argmin(losses))
            assign[i] = j
            v = ctx.train(models[j], c, ctx.S, ctx.lr)
            aggs[j].add(v - models[j], c.n_tr)
        models = [aggs[j].result(models[j]) for j in range(k)]
        hist.append(np.bincount(assign, minlength=k).tolist())
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, lambda i: models[assign[i]])
    return {"ifca": {"vecs": [models[assign[i]] for i in range(ctx.K)],
                     "comm": ctx.comm("clusters", ctx.R, k=k, extra_up=1),
                     "assign_hist": hist, "final_assign": assign.tolist()}}


def run_fedper(ctx: Ctx, w0: torch.Tensor) -> dict:
    cfg = ctx.cfg
    qffl = cfg["aggregation"] == "qffl"
    hm = ctx.hmask
    body = (~hm).to(w0.dtype)
    w = w0.clone()
    heads = [w0[hm].clone() for _ in ctx.clients]

    def full(i):
        v = w.clone()
        v[hm] = heads[i]
        return v

    for r in range(ctx.R):
        ctx.sw.start()
        agg = _aggregator(ctx)
        for i, c in enumerate(ctx.clients):
            v = full(i)
            f_k = ctx.loss_query(v, c) if qffl else None
            v_new = ctx.train(v, c, ctx.S, ctx.lr)
            heads[i] = v_new[hm].clone()
            agg.add((v_new - v) * body, f_k if qffl else c.n_tr)
        w = agg.result(w)
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, full)
    name = f"fedper_qffl_q{cfg['qffl_q']:g}{_qffl_suffix(cfg)}" if qffl else "fedper"
    return {name: {"vecs": [full(i) for i in range(ctx.K)],
                   "comm": ctx.comm("base", ctx.R, extra_up=1 if qffl else 0)}}


def _train_prox(ctx: Ctx, vec, client, steps: int, lr: float, anchor: torch.Tensor, lam: float):
    """Like Ctx.train, plus the gradient lam (w - anchor) of the mean-regularisation term. The term does not touch
    the data, so it is added after clipping and noise and costs no privacy (Liu et al. 2022, App. D.1)."""
    set_flat(ctx.model, vec)
    for _ in range(steps):
        g = ctx.grad_query(client)
        if lam > 0:
            reg = lam * (get_flat(ctx.model) - anchor)
            g = reg if g is None else g + reg
        if g is not None:
            ctx.engine.add_flat_(g, -lr)
    return get_flat(ctx.model)


def run_mrmtl(ctx: Ctx, w0: torch.Tensor) -> dict:
    """Mean-regularised multi-task learning (Liu et al., NeurIPS 2022, Alg. A1). Personal models stay on the
    clients; the server mean moves by the (data-size weighted) mean of their updates."""
    cfg = ctx.cfg
    lam = float(cfg["mrmtl_lambda"])
    w_bar = w0.clone()
    vecs = [w0.clone() for _ in ctx.clients]
    for r in range(ctx.R):
        ctx.sw.start()
        agg = MeanAgg(cfg["lr_server"])
        for i, c in enumerate(ctx.clients):
            v = _train_prox(ctx, vecs[i], c, ctx.S, ctx.lr, w_bar, lam)
            agg.add(v - vecs[i], c.n_tr)
            vecs[i] = v
        w_bar = agg.result(w_bar)
        ctx.round_times.append(ctx.sw.stop())
        if ctx.should_log(r):
            ctx.log_curve(r + 1, lambda i: vecs[i])
    return {"mrmtl": {"vecs": vecs, "comm": ctx.comm("full", ctx.R), "note": f"lambda = {lam:g}"}}


METHODS = {
    "fedavg": run_fedavg,
    "local": run_local,
    "perfedavg": run_perfedavg,
    "ifca": run_ifca,
    "fedper": run_fedper,
    "mrmtl": run_mrmtl,
}


def query_schedule(cfg: dict) -> tuple[int, int]:
    """(gradient queries, loss queries) each client answers in one run -> input to calibrate_sigma."""
    R, S, m = cfg["rounds"], cfg["local_steps"], cfg["method"]
    qffl = cfg["aggregation"] == "qffl"
    if m in ("fedavg", "fedper"):
        return R * S, (R if qffl else 0)
    if m == "local":
        if qffl:
            raise ValueError("q-FFL needs a server; not valid with method=local")
        return R * S, 0
    if m == "mrmtl":
        if qffl:
            raise ValueError("q-FFL + MR-MTL is not implemented")
        return R * S, 0
    if m == "perfedavg":
        if qffl:
            raise ValueError("q-FFL + Per-FedAvg is not implemented")
        return 2 * _meta_steps(cfg) * R + cfg["perfedavg_adapt_steps"], 0
    if m == "ifca":
        if qffl:
            raise ValueError("q-FFL + IFCA is not implemented")
        return R * S, R * cfg["ifca_k"]
    raise ValueError(f"unknown method {m}")
