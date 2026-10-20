"""Datasets, federated partitioning and per-client train/validation/test splits.

CIFAR-10 : train + test pooled (60,000 images), split across clients with a label Dirichlet(alpha)
           partition, then each client's share is split 70/10/20.
Adult    : UCI Adult (48,842 rows before cleaning); one client per occupation with >= 500 rows;
           sensitive attribute = sex (1 = male) for group-fairness gaps.
synthetic / synthetic_tab : small generated data for CPU tests (no download).

All client tensors are moved to the target device once, so training never touches a DataLoader
(important on Windows, where DataLoader workers are slow to spawn).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]


@dataclass
class Client:
    cid: int
    name: str
    x_tr: torch.Tensor
    y_tr: torch.Tensor
    x_va: torch.Tensor
    y_va: torch.Tensor
    x_te: torch.Tensor
    y_te: torch.Tensor
    s_tr: torch.Tensor | None = None
    s_va: torch.Tensor | None = None
    s_te: torch.Tensor | None = None

    @property
    def n_tr(self) -> int:
        return int(self.y_tr.shape[0])


@dataclass
class FederatedData:
    name: str
    clients: list
    num_classes: int
    input_shape: tuple
    image: bool
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    def prep(self, xb: torch.Tensor) -> torch.Tensor:
        """uint8 images -> normalised float; tabular data passes through."""
        if self.image:
            return (xb.float().div(255.0) - self.mean) / self.std
        return xb


# ----------------------------------------------------------------------------- partitioning
def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, min_size: int,
                        seed: int, max_tries: int = 2000) -> list[np.ndarray]:
    """Label-skew partition: for each class, split its indices across clients with proportions
    drawn from Dirichlet(alpha). Re-draws until every client holds at least `min_size` samples."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [rng.permutation(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"No Dirichlet partition with min size {min_size} after {max_tries} tries; "
                       "lower min_client_size or raise dirichlet_alpha.")


def split_three(idx: np.ndarray, fracs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(idx)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


# ----------------------------------------------------------------------------- raw loaders
def load_cifar10_arrays(data_dir) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    tr = CIFAR10(root=str(root), train=True, download=True)
    te = CIFAR10(root=str(root), train=False, download=True)
    x = np.concatenate([tr.data, te.data]).transpose(0, 3, 1, 2).copy()  # N, 3, 32, 32 (uint8)
    y = np.concatenate([np.asarray(tr.targets), np.asarray(te.targets)]).astype(np.int64)
    return x, y


def load_adult_frame(data_dir):
    """UCI Adult (adult.data + adult.test). Falls back to OpenML if the UCI download fails."""
    import pandas as pd

    d = Path(data_dir) / "adult"
    d.mkdir(parents=True, exist_ok=True)
    f_tr, f_te = d / "adult.data", d / "adult.test"
    if not (f_tr.exists() and f_te.exists()):
        try:
            with urllib.request.urlopen(ADULT_URL, timeout=120) as r:
                buf = r.read()
            with zipfile.ZipFile(io.BytesIO(buf)) as z:
                z.extract("adult.data", d)
                z.extract("adult.test", d)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[data] UCI download failed ({exc}); trying OpenML")
            from sklearn.datasets import fetch_openml

            df = fetch_openml("adult", version=2, as_frame=True, data_home=str(d)).frame
            df = df.rename(columns={"class": "income"})
            df = df.replace("?", np.nan).dropna()
            df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
            return df
    kw = dict(header=None, names=ADULT_COLS, skipinitialspace=True, na_values="?")
    df = pd.concat([pd.read_csv(f_tr, **kw), pd.read_csv(f_te, skiprows=1, **kw)], ignore_index=True)
    df = df.dropna()
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return df


def adult_arrays(df):
    """Features: 5 standardised numeric (capital gain/loss log-scaled) + one-hot workclass,
    marital-status, relationship, race, sex + a native-US flag. Dropped: fnlwgt (sampling weight),
    education (duplicate of education-num), occupation (it defines the client)."""
    import pandas as pd

    num = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    xn = df[num].astype(float).copy()
    xn["capital-gain"] = np.log1p(xn["capital-gain"])
    xn["capital-loss"] = np.log1p(xn["capital-loss"])
    xn = (xn - xn.mean()) / xn.std()
    cats = ["workclass", "marital-status", "relationship", "race", "sex"]
    xc = pd.get_dummies(df[cats].astype(str).apply(lambda s: s.str.strip()), dtype=float)
    us = (df["native-country"].astype(str).str.strip() == "United-States").astype(float).rename("native_us")
    x = pd.concat([xn, xc, us], axis=1).to_numpy(np.float32)
    y = df["income"].astype(str).str.startswith(">50K").to_numpy(np.int64)
    s = (df["sex"].astype(str).str.strip() == "Male").to_numpy(np.int64)
    groups = df["occupation"].astype(str).str.strip().to_numpy()
    return x, y, s, groups


def synthetic_images(n: int, num_classes: int = 10, size: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    protos = rng.normal(0.0, 1.0, (num_classes, 3, size, size))
    y = rng.integers(0, num_classes, n)
    x = protos[y] * 0.9 + rng.normal(0.0, 1.0, (n, 3, size, size))
    x = np.clip(x * 40 + 128, 0, 255).astype(np.uint8)
    return x, y.astype(np.int64)


def synthetic_tabular(n: int = 3000, d: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    groups = rng.choice(np.array(["g0", "g1", "g2", "g3", "g4"]), n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    w = rng.normal(0, 1, d)
    x = rng.normal(0, 1, (n, d)).astype(np.float32)
    x[:, 0] = s
    logits = x @ w + 0.8 * s - 0.4
    y = (logits + rng.normal(0, 1, n) > 0).astype(np.int64)
    return x, y, s.astype(np.int64), groups


# ----------------------------------------------------------------------------- builder
def build_federated_data(cfg: dict, device: torch.device) -> FederatedData:
    name = cfg["dataset"]
    fracs = cfg["split"]
    rng = np.random.default_rng(cfg["partition_seed"])
    sens = None
    if name in ("cifar10", "synthetic"):
        if name == "cifar10":
            x, y = load_cifar10_arrays(cfg["data_dir"])
        else:
            x, y = synthetic_images(cfg["num_clients"] * 200, seed=cfg["partition_seed"])
        if cfg.get("max_examples"):
            keep = rng.permutation(len(y))[: int(cfg["max_examples"])]
            x, y = x[keep], y[keep]
        parts = dirichlet_partition(y, cfg["num_clients"], cfg["dirichlet_alpha"],
                                    cfg["min_client_size"], cfg["partition_seed"])
        names = [f"client{k:02d}" for k in range(len(parts))]
        num_classes, image = 10, True
    elif name in ("adult", "synthetic_tab"):
        if name == "adult":
            x, y, sens, groups = adult_arrays(load_adult_frame(cfg["data_dir"]))
        else:
            x, y, sens, groups = synthetic_tabular(seed=cfg["partition_seed"])
        min_rows = cfg["adult_min_rows"] if name == "adult" else 100
        names = [g for g, c in Counter(groups.tolist()).most_common() if c >= min_rows]
        parts = [rng.permutation(np.flatnonzero(groups == g)) for g in names]
        num_classes, image = 2, False
    else:
        raise ValueError(f"unknown dataset {name}")

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    st = torch.from_numpy(sens) if sens is not None else None
    clients, meta_clients = [], []
    for k, idx in enumerate(parts):
        tr, va, te = (torch.from_numpy(a) for a in split_three(idx, fracs))
        c = Client(
            cid=k, name=names[k],
            x_tr=xt[tr].to(device), y_tr=yt[tr].to(device),
            x_va=xt[va].to(device), y_va=yt[va].to(device),
            x_te=xt[te].to(device), y_te=yt[te].to(device),
        )
        if st is not None:
            c.s_tr, c.s_va, c.s_te = st[tr].to(device), st[va].to(device), st[te].to(device)
        clients.append(c)
        hist = np.bincount(y[idx], minlength=num_classes).tolist()
        entry = {"name": names[k], "n": int(len(idx)), "n_train": int(len(tr)), "n_val": int(len(va)),
                 "n_test": int(len(te)), "class_hist": hist}
        if sens is not None:
            entry["female_frac"] = float(1.0 - sens[idx].mean())
            entry["positive_rate"] = float(y[idx].mean())
        meta_clients.append(entry)

    mean = std = None
    if image:
        mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    meta = {
        "dataset": name,
        "n_total": int(sum(len(p) for p in parts)),
        "num_features": int(np.prod(x.shape[1:])),
        "input_shape": list(x.shape[1:]),
        "num_classes": num_classes,
        "num_clients": len(parts),
        "clients": meta_clients,
    }
    return FederatedData(name=name, clients=clients, num_classes=num_classes,
                         input_shape=tuple(x.shape[1:]), image=image, mean=mean, std=std, meta=meta)
