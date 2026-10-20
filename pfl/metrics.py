"""Evaluation: accuracy, fairness statistics across clients, membership inference, group gaps."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- forward passes
@torch.no_grad()
def logits_of(model, x, prep, bs: int = 1024) -> torch.Tensor:
    model.eval()
    outs = [model(prep(x[i:i + bs])) for i in range(0, x.shape[0], bs)]
    return torch.cat(outs) if outs else torch.empty(0)


@torch.no_grad()
def accuracy(model, x, y, prep) -> float:
    if y.numel() == 0:
        return float("nan")
    return (logits_of(model, x, prep).argmax(1) == y).float().mean().item()


@torch.no_grad()
def record_losses(model, x, y, prep) -> torch.Tensor:
    return F.cross_entropy(logits_of(model, x, prep), y, reduction="none")


# ----------------------------------------------------------------------------- fairness
def fairness_summary(acc) -> dict:
    """Mean, worst-10% (mean of the lowest ceil(0.1 K) clients), SD, variance, Jain's index, min, max."""
    a = np.asarray([v for v in acc if not math.isnan(v)], dtype=float)
    if a.size == 0:
        return {}
    k = max(1, int(math.ceil(0.1 * a.size)))
    srt = np.sort(a)
    jain = float(a.sum() ** 2 / (a.size * (a ** 2).sum())) if (a ** 2).sum() > 0 else float("nan")
    return {
        "mean": float(a.mean()),
        "worst10": float(srt[:k].mean()),
        "std": float(a.std()),
        "var": float(a.var()),
        "jain": jain,
        "min": float(srt[0]),
        "max": float(srt[-1]),
        "k": int(a.size),
    }


# ----------------------------------------------------------------------------- membership inference
def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(score_member > score_nonmember) with ties counted half (Mann-Whitney U)."""
    from scipy.stats import rankdata

    s = np.concatenate([scores_pos, scores_neg])
    r = rankdata(s)
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def loss_threshold_mia(model, client, prep, rng: np.random.Generator) -> dict:
    """Black-box loss-threshold attack (Yeom et al. 2018).

    Members: a random subset of the client's training records. Non-members: the client's held-out
    test records. Both sides have the same size m. Score = -loss (members tend to have lower loss).
    Threshold for the accuracy/advantage numbers = mean training loss (attacker knows it)."""
    m = min(client.y_tr.shape[0], client.y_te.shape[0])
    if m < 10:
        return {"auc": float("nan"), "acc": float("nan"), "adv": float("nan"), "m": m}
    idx_in = torch.as_tensor(rng.choice(client.y_tr.shape[0], m, replace=False), device=client.y_tr.device)
    idx_out = torch.as_tensor(rng.choice(client.y_te.shape[0], m, replace=False), device=client.y_te.device)
    l_tr_all = record_losses(model, client.x_tr, client.y_tr, prep)
    l_in = l_tr_all[idx_in].cpu().numpy()
    l_out = record_losses(model, client.x_te[idx_out], client.y_te[idx_out], prep).cpu().numpy()
    tau = float(l_tr_all.mean().item())
    tpr = float((l_in <= tau).mean())
    fpr = float((l_out <= tau).mean())
    return {
        "auc": _auc(-l_in, -l_out),
        "acc": 0.5 * (tpr + (1.0 - fpr)),
        "adv": tpr - fpr,
        "m": int(m),
    }


# ----------------------------------------------------------------------------- group fairness
def group_gaps(yhat: np.ndarray, y: np.ndarray, s: np.ndarray, min_pos: int = 5) -> dict:
    """Demographic-parity gap |P(yhat=1|s=1) - P(yhat=1|s=0)| and
    equal-opportunity gap |TPR(s=1) - TPR(s=0)|. NaN when a group is too small."""
    out = {"dp_gap": float("nan"), "eo_gap": float("nan")}
    if (s == 1).sum() > 0 and (s == 0).sum() > 0:
        out["dp_gap"] = float(abs(yhat[s == 1].mean() - yhat[s == 0].mean()))
    p1, p0 = (s == 1) & (y == 1), (s == 0) & (y == 1)
    if p1.sum() >= min_pos and p0.sum() >= min_pos:
        out["eo_gap"] = float(abs(yhat[p1].mean() - yhat[p0].mean()))
    return out
