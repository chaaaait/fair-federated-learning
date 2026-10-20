"""Per-example DP-SGD and privacy accounting.

One private gradient query on client k (record-level DP, Abadi et al. 2016):
  1. Poisson-sample a batch: every training record joins independently with probability q.
  2. Compute one gradient per record (Opacus GradSampleModule, or torch.func as a fallback).
  3. Clip each per-record gradient to L2 norm <= C.
  4. Sum, add Gaussian noise N(0, sigma^2 C^2 I), divide by the expected batch size q * n_k.

Accounting: the Opacus RDP accountant composes every query a client answers. Gradient queries
are subsampled Gaussian mechanisms (rate q, noise sigma). Loss reports (q-FFL weights, IFCA
cluster choice) are Gaussian mechanisms on clipped per-record losses with noise
`loss_noise_ratio * sigma` and rate 1. `calibrate_sigma` finds the smallest sigma whose total
epsilon is <= the target, so every method lands on the same (epsilon, delta) (design default D2).
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Opacus' hooks warn that "no inputs require gradients" (expected: only parameters need grads).
warnings.filterwarnings("ignore", message="Full backward hook is firing")

# RDP orders searched by the accountant (Opacus defaults extended upwards for small epsilon).
ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))


# ----------------------------------------------------------------------------- accounting
def epsilon_from(sigma: float, q: float, t_grad: int, t_loss: int = 0, kappa: float = 30.0,
                 delta: float = 1e-5) -> float:
    from opacus.accountants import RDPAccountant

    if sigma <= 0:
        return math.inf
    acc = RDPAccountant()
    history = []
    if t_grad > 0:
        history.append((float(sigma), float(q), int(t_grad)))
    if t_loss > 0:
        history.append((float(kappa * sigma), 1.0, int(t_loss)))
    acc.history = history
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "optimal order is the largest/smallest alpha" at bisection extremes
        return float(acc.get_epsilon(delta=delta, alphas=ALPHAS))


def calibrate_sigma(target_eps: float, q: float, t_grad: int, t_loss: int = 0, kappa: float = 30.0,
                    delta: float = 1e-5, lo: float = 0.2, hi: float = 400.0, tol: float = 1e-4) -> float:
    """Smallest noise multiplier with epsilon <= target (geometric bisection; epsilon falls as sigma grows)."""
    if epsilon_from(hi, q, t_grad, t_loss, kappa, delta) > target_eps:
        raise ValueError(f"target epsilon {target_eps} unreachable even with sigma={hi}")
    if epsilon_from(lo, q, t_grad, t_loss, kappa, delta) <= target_eps:
        return lo
    while hi / lo > 1.0 + tol:
        mid = math.sqrt(lo * hi)
        if epsilon_from(mid, q, t_grad, t_loss, kappa, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------- gradients
class GradEngine:
    """Computes plain or private gradients for `model` as one flat vector (parameter order of model.parameters())."""

    def __init__(self, model: nn.Module, backend: str = "opacus", max_physical_batch: int = 128):
        self.model = model
        self.params = list(model.parameters())
        self.P = sum(p.numel() for p in self.params)
        self.backend = backend
        self.max_physical_batch = max_physical_batch
        self.gsm = None
        if backend == "opacus":
            from opacus import GradSampleModule

            self.gsm = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            self.gsm.disable_hooks()
        elif backend != "functorch":
            raise ValueError(f"unknown grad backend {backend}")

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def poisson_indices(n: int, q: float, gen: torch.Generator, device) -> torch.Tensor:
        mask = torch.rand(n, generator=gen, device=device) < q
        return mask.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def add_flat_(self, vec: torch.Tensor, alpha: float) -> None:
        o = 0
        for p in self.params:
            n = p.numel()
            p.add_(vec[o:o + n].view_as(p), alpha=alpha)
            o += n

    def per_sample_grads(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """[B, P] matrix of per-example gradients of the cross-entropy loss."""
        b = xb.shape[0]
        self.model.train()
        if self.backend == "opacus":
            for p in self.params:
                p.grad = None
                p.grad_sample = None
            self.gsm.enable_hooks()
            try:
                out = self.gsm(xb)
                F.cross_entropy(out, yb, reduction="sum").backward()
                g = torch.cat([p.grad_sample.reshape(b, -1) for p in self.params], dim=1)
            finally:
                self.gsm.disable_hooks()
                for p in self.params:
                    p.grad = None
                    p.grad_sample = None
            return g
        from torch.func import functional_call, grad, vmap

        names = [n for n, _ in self.model.named_parameters()]
        params = {n: p.detach() for n, p in self.model.named_parameters()}

        def loss_one(prm, x, y):
            out = functional_call(self.model, prm, (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))

        g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, xb, yb)
        return torch.cat([g[n].reshape(b, -1) for n in names], dim=1)

    # -- queries ---------------------------------------------------------------------------
    def private_grad(self, x, y, prep, q: float, sigma: float, clip: float, gen: torch.Generator) -> torch.Tensor:
        """One noisy gradient query. Always returns a vector (noise only if the batch is empty)."""
        n = y.shape[0]
        idx = self.poisson_indices(n, q, gen, y.device)
        total = torch.zeros(self.P, device=y.device)
        for chunk in idx.split(self.max_physical_batch):
            g = self.per_sample_grads(prep(x[chunk]), y[chunk])
            factor = (clip / (g.norm(dim=1) + 1e-6)).clamp(max=1.0)
            total += (factor.unsqueeze(1) * g).sum(dim=0)
        if sigma > 0:
            total += torch.randn(self.P, generator=gen, device=y.device) * (sigma * clip)
        return total / max(q * n, 1e-12)

    def plain_grad(self, x, y, prep, q: float, gen: torch.Generator):
        """Non-private minibatch gradient with the same Poisson sampling; None if the batch is empty."""
        idx = self.poisson_indices(y.shape[0], q, gen, y.device)
        if idx.numel() == 0:
            return None
        self.model.train()
        loss = F.cross_entropy(self.model(prep(x[idx])), y[idx])
        grads = torch.autograd.grad(loss, self.params)
        return torch.cat([g.reshape(-1) for g in grads])
