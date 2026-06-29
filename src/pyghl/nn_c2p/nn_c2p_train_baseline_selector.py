"""Train a ranking-aware abstaining selector over Noble2D baseline guesses."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TIE_BREAK_ORDER = {
    "x_lo": 0,
    "phase_a": 1,
    "midpoint": 2,
    "x_hi": 3,
}

REFERENCE_PHASE_A_SCORE = 27.57
REFERENCE_ORACLE_SCORE = 21.13
EXPERIMENT11_PHASE_A_SCORE = 28.4386
EXPERIMENT11_PHASE_A_FAILURE = 0.1507
EXPERIMENT11_ORACLE_SCORE = 21.9810
EXPERIMENT11_ORACLE_FAILURE = 0.1107
EXPERIMENT11_FULL_CLASSIFIER_SCORE = 25.1035
EXPERIMENT11_FULL_CLASSIFIER_FAILURE = 0.1400
EXPERIMENT11_PAIR_ABSTAIN_SCORE = 26.0951
EXPERIMENT11_PAIR_ABSTAIN_FAILURE = 0.1487
FAILURE_SCORE = 100.0


@dataclass
class BaselineData:
    names: list[str]
    q: np.ndarray
    r: np.ndarray
    s: np.ndarray
    t: np.ndarray
    x_lo: np.ndarray
    x_hi: np.ndarray
    y: np.ndarray
    success: np.ndarray
    n_iter: np.ndarray
    cons_error: np.ndarray
    residual_norm: np.ndarray | None
    failure_code: np.ndarray | None
    state_type: np.ndarray | None
    x_exact_success: np.ndarray | None
    rho: np.ndarray | None
    temp: np.ndarray | None
    ye: np.ndarray | None
    w_lorentz: np.ndarray | None
    log_pmagop: np.ndarray | None


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairSelector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        self.body = MLP(in_dim, 2, hidden_dim, depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(x)
        success_logit = out[..., 0]
        iter_pred = torch.nn.functional.softplus(out[..., 1]) + 1.0
        pred_cost = 100.0 * torch.sigmoid(-success_logit) + iter_pred
        return success_logit, iter_pred, pred_cost


class PairwiseComparator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        self.body = MLP(in_dim, 1, hidden_dim, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)


def _decode_names(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def _read_optional(h5: h5py.File, name: str):
    return h5[name][()] if name in h5 else None


def _default_label_file(cost_map_file: Path) -> Path | None:
    candidate = Path(str(cost_map_file).replace("cost_maps", "solver_labels"))
    return candidate if candidate.exists() else None


def load_cost_map_baseline_data(
    cost_map_file: str | Path,
    candidate_names: list[str],
    label_file: str | Path | None,
    limit_states: int | None = None,
) -> BaselineData:
    with h5py.File(cost_map_file, "r") as h5:
        all_names = _decode_names(h5["baselines/names"][()])
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        missing = [name for name in candidate_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Missing requested baseline candidates: {missing}; have {all_names}")
        idx = np.array([name_to_idx[name] for name in candidate_names], dtype=np.int64)
        sl = slice(None if limit_states is None else limit_states)
        x_exact_success = None
        if "x_exact" in name_to_idx:
            x_exact_success = h5["baselines/success"][sl][:, name_to_idx["x_exact"]].astype(bool)

        data = BaselineData(
            names=candidate_names,
            q=h5["states/q"][sl].astype(np.float64),
            r=h5["states/r"][sl].astype(np.float64),
            s=h5["states/s"][sl].astype(np.float64),
            t=h5["states/t"][sl].astype(np.float64),
            x_lo=h5["states/x_lo"][sl].astype(np.float64),
            x_hi=h5["states/x_hi"][sl].astype(np.float64),
            y=h5["baselines/y"][sl][:, idx].astype(np.float64),
            success=h5["baselines/success"][sl][:, idx].astype(bool),
            n_iter=h5["baselines/n_iter"][sl][:, idx].astype(np.float64),
            cons_error=h5["baselines/cons_error"][sl][:, idx].astype(np.float64),
            residual_norm=(
                h5["baselines/residual_norm"][sl][:, idx].astype(np.float64)
                if "baselines/residual_norm" in h5
                else None
            ),
            failure_code=(
                h5["baselines/failure_code"][sl][:, idx].astype(np.int64)
                if "baselines/failure_code" in h5
                else None
            ),
            state_type=None,
            x_exact_success=x_exact_success,
            rho=_read_optional(h5, "states/rho"),
            temp=_read_optional(h5, "states/T"),
            ye=_read_optional(h5, "states/Ye"),
            w_lorentz=_read_optional(h5, "states/W"),
            log_pmagop=_read_optional(h5, "states/log_PmagoP"),
        )

    if limit_states is not None:
        for attr in ("rho", "temp", "ye", "w_lorentz", "log_pmagop"):
            arr = getattr(data, attr)
            if arr is not None:
                setattr(data, attr, arr[:limit_states].astype(np.float64))

    label_path = Path(label_file) if label_file is not None else _default_label_file(Path(cost_map_file))
    if label_path is not None and label_path.exists():
        with h5py.File(label_path, "r") as h5:
            if "labels/state_type" in h5:
                data.state_type = h5["labels/state_type"][: len(data.q)]

    return data


def rank_candidates_for_state(
    names: list[str],
    success: np.ndarray,
    n_iter: np.ndarray,
    cons_error: np.ndarray,
    residual_norm: np.ndarray | None = None,
    failure_code: np.ndarray | None = None,
) -> np.ndarray:
    keys = []
    for i, name in enumerate(names):
        ok = bool(success[i])
        iter_key = float(n_iter[i]) if ok and np.isfinite(n_iter[i]) and n_iter[i] >= 0 else 1e9
        err_key = float(cons_error[i]) if ok and np.isfinite(cons_error[i]) else 1e30
        res_key = 1e30
        if residual_norm is not None and np.isfinite(residual_norm[i]):
            res_key = float(residual_norm[i])
        fail_key = 0
        if not ok and failure_code is not None:
            fail_key = abs(int(failure_code[i]))
        keys.append(
            (
                0 if ok else 1,
                iter_key,
                err_key,
                res_key,
                fail_key,
                TIE_BREAK_ORDER.get(name, 100 + i),
            )
        )
    return np.array(sorted(range(len(names)), key=lambda i: keys[i]), dtype=np.int64)


def build_oracle_labels(data: BaselineData) -> tuple[np.ndarray, np.ndarray]:
    n = len(data.q)
    ranks = np.zeros((n, len(data.names)), dtype=np.int64)
    best = np.zeros(n, dtype=np.int64)
    for i in range(n):
        order = rank_candidates_for_state(
            data.names,
            data.success[i],
            data.n_iter[i],
            data.cons_error[i],
            data.residual_norm[i] if data.residual_norm is not None else None,
            data.failure_code[i] if data.failure_code is not None else None,
        )
        ranks[i, order] = np.arange(len(data.names), dtype=np.int64)
        best[i] = order[0]
    return best, ranks


def build_state_splits(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def _normalize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x - mean) / std


def make_minimal_classifier_features(data: BaselineData) -> np.ndarray:
    return np.column_stack((data.q, data.r, data.s, data.t)).astype(np.float32)


def _phase_a_index(names: list[str]) -> int | None:
    return names.index("phase_a") if "phase_a" in names else None


def make_full_classifier_features(data: BaselineData) -> np.ndarray:
    phase_idx = _phase_a_index(data.names)
    phase_y = np.zeros_like(data.q)
    phase_avail = np.zeros_like(data.q)
    if phase_idx is not None:
        phase_y = np.nan_to_num(data.y[:, phase_idx], nan=0.0, posinf=0.0, neginf=0.0)
        phase_avail = np.isfinite(data.y[:, phase_idx]).astype(np.float64)
    width = data.x_hi - data.x_lo
    return np.column_stack(
        (data.q, data.r, data.s, data.t, data.x_lo, data.x_hi, width, phase_y, phase_avail)
    ).astype(np.float32)


def make_pair_features(data: BaselineData) -> np.ndarray:
    base = make_full_classifier_features(data).astype(np.float64)
    n_state = len(data.q)
    n_cand = len(data.names)
    out = []
    phase_avail = base[:, -1]
    for j, name in enumerate(data.names):
        one_hot = np.zeros((n_state, n_cand), dtype=np.float64)
        one_hot[:, j] = 1.0
        candidate_y = np.nan_to_num(data.y[:, j], nan=0.0, posinf=0.0, neginf=0.0)[:, None]
        is_boundary = ((candidate_y[:, 0] <= 1e-12) | (candidate_y[:, 0] >= 1.0 - 1e-12)).astype(
            np.float64
        )[:, None]
        flags = np.column_stack(
            (
                is_boundary[:, 0],
                np.full(n_state, 1.0 if name == "x_lo" else 0.0),
                np.full(n_state, 1.0 if name == "x_hi" else 0.0),
                np.full(n_state, 1.0 if name == "midpoint" else 0.0),
                np.full(n_state, 1.0 if name == "phase_a" else 0.0),
                phase_avail,
            )
        )
        out.append(np.column_stack((base, candidate_y, one_hot, flags)))
    return np.stack(out, axis=1).astype(np.float32)


def _class_weights(labels: np.ndarray, n_class: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_class).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights *= n_class / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def _best_state_score(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(x_val), y_val)
    return float(loss.item())


def train_hard_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> tuple[MLP, dict]:
    mean, std = _normalize_fit(x[train_idx])
    x_norm = _normalize_apply(x, mean, std).astype(np.float32)
    model = MLP(x.shape[1], int(np.max(y)) + 1, hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = _class_weights(y[train_idx], model.net[-1].out_features)
    train_ds = TensorDataset(torch.tensor(x_norm[train_idx]), torch.tensor(y[train_idx], dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_val = torch.tensor(x_norm[val_idx])
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb, weight=weights)
            loss.backward()
            opt.step()
        val_loss = _best_state_score(model, x_val, y_val)
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_val_loss": best_loss}


def _multiclass_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    variant: str,
    soft_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if variant == "focal":
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
        pt = torch.softmax(logits, dim=1)[torch.arange(len(labels), device=labels.device), labels]
        return (((1.0 - pt) ** 2.0) * ce).mean()
    if variant == "soft_regret":
        if soft_targets is None:
            raise ValueError("soft_regret requires soft targets")
        logp = torch.log_softmax(logits, dim=1)
        return -(soft_targets * logp).sum(dim=1).mean()
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights)


def train_multiclass_selector(
    x: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    variant: str,
    soft_targets: np.ndarray | None = None,
) -> tuple[MLP, dict]:
    mean, std = _normalize_fit(x[train_idx])
    x_norm = _normalize_apply(x, mean, std).astype(np.float32)
    model = MLP(x.shape[1], int(np.max(labels)) + 1, hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = _class_weights(labels[train_idx], model.net[-1].out_features)
    train_items = [
        torch.tensor(x_norm[train_idx]),
        torch.tensor(labels[train_idx], dtype=torch.long),
    ]
    val_items = [
        torch.tensor(x_norm[val_idx]),
        torch.tensor(labels[val_idx], dtype=torch.long),
    ]
    if soft_targets is not None:
        train_items.append(torch.tensor(soft_targets[train_idx], dtype=torch.float32))
        val_items.append(torch.tensor(soft_targets[val_idx], dtype=torch.float32))
    train_dl = DataLoader(TensorDataset(*train_items), batch_size=batch_size, shuffle=True)
    x_val = val_items[0]
    y_val = val_items[1]
    ysoft_val = val_items[2] if len(val_items) > 2 else None
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            xb, yb = batch[0], batch[1]
            ysoft = batch[2] if len(batch) > 2 else None
            opt.zero_grad(set_to_none=True)
            loss = _multiclass_loss(model(xb), yb, weights, variant, ysoft)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = _multiclass_loss(model(x_val), y_val, weights, variant, ysoft_val).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_val_loss": best_loss, "variant": variant}


def _pair_loss(
    success_logit: torch.Tensor,
    iter_pred: torch.Tensor,
    pred_cost: torch.Tensor,
    success: torch.Tensor,
    n_iter: torch.Tensor,
    ranks: torch.Tensor,
    success_weight: float,
    iter_weight: float,
    ranking_weight: float,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(success_logit, success.float())
    ok = success.bool()
    if ok.any():
        iter_loss = torch.nn.functional.mse_loss(iter_pred[ok], n_iter[ok].float())
    else:
        iter_loss = torch.zeros((), dtype=pred_cost.dtype, device=pred_cost.device)
    rank_loss = torch.zeros((), dtype=pred_cost.dtype, device=pred_cost.device)
    n_pairs = 0
    n_cand = pred_cost.shape[1]
    for a in range(n_cand):
        for b in range(a + 1, n_cand):
            a_better = ranks[:, a] < ranks[:, b]
            better = torch.where(a_better, pred_cost[:, a], pred_cost[:, b])
            worse = torch.where(a_better, pred_cost[:, b], pred_cost[:, a])
            rank_loss = rank_loss + torch.relu(1.0 + better - worse).mean()
            n_pairs += 1
    rank_loss = rank_loss / max(n_pairs, 1)
    return success_weight * bce + iter_weight * iter_loss + ranking_weight * rank_loss


def _weighted_pair_loss(
    success_logit: torch.Tensor,
    iter_pred: torch.Tensor,
    pred_cost: torch.Tensor,
    success: torch.Tensor,
    n_iter: torch.Tensor,
    ranks: torch.Tensor,
    winner: torch.Tensor,
    winner_weights: torch.Tensor,
    true_cost: torch.Tensor,
    success_weight: float,
    iter_weight: float,
    ranking_weight: float,
    rank_margin: float,
    failure_penalty: float,
) -> torch.Tensor:
    bce_terms = torch.nn.functional.binary_cross_entropy_with_logits(
        success_logit, success.float(), reduction="none"
    )
    bce = bce_terms.mean(dim=0).mean()
    ok = success.bool()
    if ok.any():
        iter_loss = torch.nn.functional.mse_loss(iter_pred[ok], n_iter[ok].float())
    else:
        iter_loss = torch.zeros((), dtype=pred_cost.dtype, device=pred_cost.device)
    all_failed = ~success.bool().any(dim=1)
    state_w = winner_weights[winner].to(pred_cost.device)
    state_w = torch.where(all_failed, 0.1 * state_w, state_w)
    rank_loss = torch.zeros((), dtype=pred_cost.dtype, device=pred_cost.device)
    weight_sum = torch.zeros((), dtype=pred_cost.dtype, device=pred_cost.device)
    n_cand = pred_cost.shape[1]
    for a in range(n_cand):
        for b in range(a + 1, n_cand):
            a_better = ranks[:, a] < ranks[:, b]
            better = torch.where(a_better, pred_cost[:, a], pred_cost[:, b])
            worse = torch.where(a_better, pred_cost[:, b], pred_cost[:, a])
            regret_w = 1.0 + torch.clamp(torch.abs(true_cost[:, a] - true_cost[:, b]), max=20.0)
            pair_w = state_w * regret_w
            terms = torch.relu(rank_margin + better - worse) * pair_w
            rank_loss = rank_loss + terms.sum()
            weight_sum = weight_sum + pair_w.sum()
    rank_loss = rank_loss / torch.clamp(weight_sum, min=1.0)
    return success_weight * bce + iter_weight * iter_loss + ranking_weight * rank_loss


def train_pair_selector(
    pair_x: np.ndarray,
    success: np.ndarray,
    n_iter: np.ndarray,
    ranks: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    success_weight: float,
    iter_weight: float,
    ranking_weight: float,
) -> tuple[PairSelector, dict]:
    train_flat = pair_x[train_idx].reshape(-1, pair_x.shape[-1])
    mean, std = _normalize_fit(train_flat)
    pair_norm = _normalize_apply(pair_x.reshape(-1, pair_x.shape[-1]), mean, std).reshape(pair_x.shape)
    model = PairSelector(pair_x.shape[-1], hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = TensorDataset(
        torch.tensor(pair_norm[train_idx]),
        torch.tensor(success[train_idx]),
        torch.tensor(n_iter[train_idx]),
        torch.tensor(ranks[train_idx], dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_tensors = (
        torch.tensor(pair_norm[val_idx]),
        torch.tensor(success[val_idx]),
        torch.tensor(n_iter[val_idx]),
        torch.tensor(ranks[val_idx], dtype=torch.long),
    )
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, sb, nb, rb in train_dl:
            opt.zero_grad(set_to_none=True)
            slogit, ipred, pcost = model(xb)
            loss = _pair_loss(
                slogit,
                ipred,
                pcost,
                sb,
                nb,
                rb,
                success_weight,
                iter_weight,
                ranking_weight,
            )
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            slogit, ipred, pcost = model(val_tensors[0])
            val_loss = _pair_loss(
                slogit,
                ipred,
                pcost,
                val_tensors[1],
                val_tensors[2],
                val_tensors[3],
                success_weight,
                iter_weight,
                ranking_weight,
            ).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_val_loss": best_loss}


def _winner_weights(best: np.ndarray, train_idx: np.ndarray, n_class: int, max_weight: float) -> torch.Tensor:
    counts = np.bincount(best[train_idx], minlength=n_class).astype(np.float64)
    weights = len(train_idx) / np.maximum(counts * n_class, 1.0)
    weights = np.minimum(weights, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def train_weighted_pair_selector(
    pair_x: np.ndarray,
    success: np.ndarray,
    n_iter: np.ndarray,
    ranks: np.ndarray,
    best: np.ndarray,
    true_cost: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    success_weight: float,
    iter_weight: float,
    ranking_weight: float,
    rank_margin: float,
    failure_penalty: float,
    winner_max_weight: float,
) -> tuple[PairSelector, dict]:
    train_flat = pair_x[train_idx].reshape(-1, pair_x.shape[-1])
    mean, std = _normalize_fit(train_flat)
    pair_norm = _normalize_apply(pair_x.reshape(-1, pair_x.shape[-1]), mean, std).reshape(pair_x.shape)
    model = PairSelector(pair_x.shape[-1], hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    winner_weights = _winner_weights(best, train_idx, pair_x.shape[1], winner_max_weight)
    train_ds = TensorDataset(
        torch.tensor(pair_norm[train_idx]),
        torch.tensor(success[train_idx]),
        torch.tensor(n_iter[train_idx]),
        torch.tensor(ranks[train_idx], dtype=torch.long),
        torch.tensor(best[train_idx], dtype=torch.long),
        torch.tensor(true_cost[train_idx], dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_tensors = (
        torch.tensor(pair_norm[val_idx]),
        torch.tensor(success[val_idx]),
        torch.tensor(n_iter[val_idx]),
        torch.tensor(ranks[val_idx], dtype=torch.long),
        torch.tensor(best[val_idx], dtype=torch.long),
        torch.tensor(true_cost[val_idx], dtype=torch.float32),
    )
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, sb, nb, rb, wb, cb in train_dl:
            opt.zero_grad(set_to_none=True)
            slogit, ipred, pcost = model(xb)
            loss = _weighted_pair_loss(
                slogit,
                ipred,
                pcost,
                sb,
                nb,
                rb,
                wb,
                winner_weights,
                cb,
                success_weight,
                iter_weight,
                ranking_weight,
                rank_margin,
                failure_penalty,
            )
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            slogit, ipred, pcost = model(val_tensors[0])
            val_loss = _weighted_pair_loss(
                slogit,
                ipred,
                pcost,
                val_tensors[1],
                val_tensors[2],
                val_tensors[3],
                val_tensors[4],
                winner_weights,
                val_tensors[5],
                success_weight,
                iter_weight,
                ranking_weight,
                rank_margin,
                failure_penalty,
            ).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_val_loss": best_loss, "winner_weights": winner_weights.numpy()}


def predict_classifier(model: MLP, norm: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_norm = _normalize_apply(x, norm["mean"], norm["std"]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_norm))
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    return np.argmax(prob, axis=1), prob


def predict_pair_costs(model: PairSelector, norm: dict, pair_x: np.ndarray) -> np.ndarray:
    shape = pair_x.shape
    flat = _normalize_apply(pair_x.reshape(-1, shape[-1]), norm["mean"], norm["std"]).reshape(shape)
    model.eval()
    with torch.no_grad():
        _, _, pred_cost = model(torch.tensor(flat.astype(np.float32)))
    return pred_cost.cpu().numpy()


def predict_pair_details(model: PairSelector, norm: dict, pair_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = pair_x.shape
    flat = _normalize_apply(pair_x.reshape(-1, shape[-1]), norm["mean"], norm["std"]).reshape(shape)
    model.eval()
    with torch.no_grad():
        success_logit, iter_pred, pred_cost = model(torch.tensor(flat.astype(np.float32)))
        psuccess = torch.sigmoid(success_logit)
    return psuccess.cpu().numpy(), iter_pred.cpu().numpy(), pred_cost.cpu().numpy()


def make_pairwise_comparison_features(pair_x: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = [(a, b) for a in range(pair_x.shape[1]) for b in range(a + 1, pair_x.shape[1])]
    feats = []
    for a, b in pairs:
        xa = pair_x[:, a, :]
        xb = pair_x[:, b, :]
        feats.append(np.concatenate((xa, xb, xa - xb), axis=1))
    return np.stack(feats, axis=1).astype(np.float32), pairs


def _pairwise_compare_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return (terms * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def train_pairwise_comparator(
    cmp_x: np.ndarray,
    pairs: list[tuple[int, int]],
    ranks: np.ndarray,
    best: np.ndarray,
    true_cost: np.ndarray,
    success: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    winner_max_weight: float,
) -> tuple[PairwiseComparator, dict]:
    labels = np.zeros((len(best), len(pairs)), dtype=np.float32)
    weights = np.zeros_like(labels)
    n_cand = max(max(a, b) for a, b in pairs) + 1
    ww = _winner_weights(best, train_idx, n_cand, winner_max_weight).numpy()
    all_failed = ~np.any(success, axis=1)
    for p, (a, b) in enumerate(pairs):
        labels[:, p] = (ranks[:, a] < ranks[:, b]).astype(np.float32)
        regret_w = 1.0 + np.minimum(20.0, np.abs(true_cost[:, a] - true_cost[:, b]))
        state_w = ww[best] * regret_w
        state_w[all_failed] *= 0.1
        weights[:, p] = state_w.astype(np.float32)
    train_flat = cmp_x[train_idx].reshape(-1, cmp_x.shape[-1])
    mean, std = _normalize_fit(train_flat)
    cmp_norm = _normalize_apply(cmp_x.reshape(-1, cmp_x.shape[-1]), mean, std).reshape(cmp_x.shape)
    model = PairwiseComparator(cmp_x.shape[-1], hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = TensorDataset(
        torch.tensor(cmp_norm[train_idx].reshape(-1, cmp_x.shape[-1])),
        torch.tensor(labels[train_idx].reshape(-1)),
        torch.tensor(weights[train_idx].reshape(-1)),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_val = torch.tensor(cmp_norm[val_idx].reshape(-1, cmp_x.shape[-1]))
    y_val = torch.tensor(labels[val_idx].reshape(-1))
    w_val = torch.tensor(weights[val_idx].reshape(-1))
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = _pairwise_compare_loss(model(xb), yb, wb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = _pairwise_compare_loss(model(x_val), y_val, w_val).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_val_loss": best_loss, "pairs": pairs}


def predict_pairwise_scores(
    model: PairwiseComparator,
    norm: dict,
    cmp_x: np.ndarray,
    n_cand: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = cmp_x.shape
    flat = _normalize_apply(cmp_x.reshape(-1, shape[-1]), norm["mean"], norm["std"]).reshape(shape)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(flat.astype(np.float32)).reshape(-1, shape[-1])))
    prob = prob.cpu().numpy().reshape(shape[0], shape[1])
    borda = np.zeros((shape[0], n_cand), dtype=np.float64)
    copeland = np.zeros_like(borda)
    for p, (a, b) in enumerate(norm["pairs"]):
        borda[:, a] += prob[:, p]
        borda[:, b] += 1.0 - prob[:, p]
        copeland[:, a] += prob[:, p] > 0.5
        copeland[:, b] += prob[:, p] <= 0.5
    return borda, copeland


def per_sample_cost(success: np.ndarray, n_iter: np.ndarray) -> np.ndarray:
    return np.where(success, n_iter, FAILURE_SCORE).astype(np.float64)


def all_candidate_costs(data: BaselineData, failure_penalty: float) -> np.ndarray:
    return np.where(data.success, data.n_iter, failure_penalty).astype(np.float64)


def make_soft_regret_targets(costs: np.ndarray, temperature: float) -> np.ndarray:
    regret = costs - np.min(costs, axis=1, keepdims=True)
    logits = -regret / max(temperature, 1e-12)
    logits -= np.max(logits, axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= np.sum(prob, axis=1, keepdims=True)
    return prob.astype(np.float32)


def evaluate_policy(
    name: str,
    chosen: np.ndarray,
    idx: np.ndarray,
    data: BaselineData,
    best: np.ndarray,
) -> dict:
    chosen = np.asarray(chosen, dtype=np.int64)
    row = np.arange(len(chosen))
    success = data.success[idx][row, chosen]
    nit = data.n_iter[idx][row, chosen]
    err = data.cons_error[idx][row, chosen]
    ok_nit = nit[success]
    ok_err = err[success & np.isfinite(err)]
    if ok_nit.size:
        mean_iter = float(np.mean(ok_nit))
        median_iter = float(np.median(ok_nit))
        p90 = float(np.percentile(ok_nit, 90))
        p95 = float(np.percentile(ok_nit, 95))
        p99 = float(np.percentile(ok_nit, 99))
        max_iter = float(np.max(ok_nit))
    else:
        mean_iter = median_iter = p90 = p95 = p99 = max_iter = math.nan
    failure_rate = float(1.0 - np.mean(success))
    score = 100.0 * failure_rate
    if ok_nit.size:
        score += mean_iter + 0.25 * p95 + 0.10 * p99
    else:
        score += FAILURE_SCORE
    return {
        "name": name,
        "n": int(len(idx)),
        "failure_rate": failure_rate,
        "success_rate": float(np.mean(success)),
        "mean_iter": mean_iter,
        "median_iter": median_iter,
        "p90_iter": p90,
        "p95_iter": p95,
        "p99_iter": p99,
        "max_iter": max_iter,
        "score": float(score),
        "cons_error_mean": float(np.mean(ok_err)) if ok_err.size else math.nan,
        "cons_error_p95": float(np.percentile(ok_err, 95)) if ok_err.size else math.nan,
        "cons_error_p99": float(np.percentile(ok_err, 99)) if ok_err.size else math.nan,
        "chosen_frequency": np.bincount(chosen, minlength=len(data.names)).tolist(),
        "oracle_accuracy": float(np.mean(chosen == best[idx])),
    }


def evaluate_abstaining_policy(
    name: str,
    pred_costs: np.ndarray,
    fallback_id: int,
    margin: float,
    idx: np.ndarray,
    data: BaselineData,
    best: np.ndarray,
    ranks: np.ndarray,
) -> tuple[dict, np.ndarray]:
    selected = np.argmin(pred_costs, axis=1)
    gain = pred_costs[np.arange(len(idx)), fallback_id] - pred_costs[np.arange(len(idx)), selected]
    override = gain >= margin
    chosen = np.where(override, selected, fallback_id)
    metrics = evaluate_policy(name, chosen, idx, data, best)
    fallback_cost = per_sample_cost(data.success[idx, fallback_id], data.n_iter[idx, fallback_id])
    chosen_cost = per_sample_cost(data.success[idx][np.arange(len(idx)), chosen], data.n_iter[idx][np.arange(len(idx)), chosen])
    override_win = override & (ranks[idx][np.arange(len(idx)), chosen] < ranks[idx, fallback_id])
    override_loss = override & (ranks[idx][np.arange(len(idx)), chosen] > ranks[idx, fallback_id])
    metrics.update(
        {
            "fallback": data.names[fallback_id],
            "margin": float(margin),
            "override_rate": float(np.mean(override)),
            "override_win_rate": float(np.mean(override_win[override])) if np.any(override) else 0.0,
            "override_loss_rate": float(np.mean(override_loss[override])) if np.any(override) else 0.0,
            "avg_gain_when_win": float(np.mean(fallback_cost[override_win] - chosen_cost[override_win]))
            if np.any(override_win)
            else 0.0,
            "avg_loss_when_loss": float(np.mean(chosen_cost[override_loss] - fallback_cost[override_loss]))
            if np.any(override_loss)
            else 0.0,
            "n_override": int(np.sum(override)),
        }
    )
    return metrics, chosen


def evaluate_score_abstaining_policy(
    name: str,
    scores: np.ndarray,
    fallback_id: int,
    margin: float,
    idx: np.ndarray,
    data: BaselineData,
    best: np.ndarray,
    ranks: np.ndarray,
    higher_is_better: bool,
) -> tuple[dict, np.ndarray]:
    selected = np.argmax(scores, axis=1) if higher_is_better else np.argmin(scores, axis=1)
    if higher_is_better:
        gain = scores[np.arange(len(idx)), selected] - scores[np.arange(len(idx)), fallback_id]
    else:
        gain = scores[np.arange(len(idx)), fallback_id] - scores[np.arange(len(idx)), selected]
    override = gain >= margin
    chosen = np.where(override, selected, fallback_id)
    metrics = evaluate_policy(name, chosen, idx, data, best)
    fallback_cost = per_sample_cost(data.success[idx, fallback_id], data.n_iter[idx, fallback_id])
    chosen_cost = per_sample_cost(data.success[idx][np.arange(len(idx)), chosen], data.n_iter[idx][np.arange(len(idx)), chosen])
    override_win = override & (ranks[idx][np.arange(len(idx)), chosen] < ranks[idx, fallback_id])
    override_loss = override & (ranks[idx][np.arange(len(idx)), chosen] > ranks[idx, fallback_id])
    metrics.update(
        {
            "fallback": data.names[fallback_id],
            "margin": float(margin),
            "override_rate": float(np.mean(override)),
            "override_win_rate": float(np.mean(override_win[override])) if np.any(override) else 0.0,
            "override_loss_rate": float(np.mean(override_loss[override])) if np.any(override) else 0.0,
            "avg_gain_when_win": float(np.mean(fallback_cost[override_win] - chosen_cost[override_win]))
            if np.any(override_win)
            else 0.0,
            "avg_loss_when_loss": float(np.mean(chosen_cost[override_loss] - fallback_cost[override_loss]))
            if np.any(override_loss)
            else 0.0,
            "n_override": int(np.sum(override)),
        }
    )
    return metrics, chosen


def confusion_matrix(pred: np.ndarray, truth: np.ndarray, n_class: int) -> np.ndarray:
    out = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(truth, pred, strict=False):
        out[int(t), int(p)] += 1
    return out


def _fmt_metrics(m: dict) -> str:
    return (
        f"{m['name']:36s} fail={m['failure_rate']:.4f} "
        f"score={m['score']:.4f} mean={m['mean_iter']:.4f} "
        f"p95={m['p95_iter']:.4f} p99={m['p99_iter']:.4f} "
        f"err95={m['cons_error_p95']:.3e} oracle_acc={m['oracle_accuracy']:.4f}"
    )


def _describe_counts(labels: np.ndarray, names: list[str]) -> str:
    counts = np.bincount(labels, minlength=len(names))
    return ", ".join(f"{name}:{counts[i]}" for i, name in enumerate(names))


def _split_audit(data: BaselineData, best: np.ndarray, splits: dict[str, np.ndarray]) -> list[str]:
    lines = ["Split audit:"]
    for split, idx in splits.items():
        lines.append(f"  {split}: n={len(idx)} best=({_describe_counts(best[idx], data.names)})")
        if data.x_exact_success is not None:
            lines.append(f"    x_exact_success_rate: {np.mean(data.x_exact_success[idx]):.4f}")
        if data.state_type is not None:
            vals, counts = np.unique(data.state_type[idx], return_counts=True)
            lines.append("    state_type: " + ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)))
        for label, arr in (
            ("rho", data.rho),
            ("T", data.temp),
            ("Ye", data.ye),
            ("W", data.w_lorentz),
            ("log_PmagoP", data.log_pmagop),
            ("q", data.q),
            ("r", data.r),
            ("s", data.s),
            ("t", data.t),
        ):
            if arr is None:
                continue
            vals = np.asarray(arr)[idx]
            lines.append(
                f"    {label}: mean={np.nanmean(vals):.4e} p05={np.nanpercentile(vals,5):.4e} "
                f"p95={np.nanpercentile(vals,95):.4e}"
            )
    return lines


def _subset_diagnostics(
    data: BaselineData,
    idx: np.ndarray,
    chosen: np.ndarray,
    best: np.ndarray,
    title: str,
) -> list[str]:
    lines = [f"Subset diagnostics for {title}:"]
    name_to_idx = {name: i for i, name in enumerate(data.names)}
    masks: list[tuple[str, np.ndarray]] = []
    for i, name in enumerate(data.names):
        masks.append((f"oracle_{name}_wins", best[idx] == i))
    if "phase_a" in name_to_idx and "x_lo" in name_to_idx:
        pa = name_to_idx["phase_a"]
        lo = name_to_idx["x_lo"]
        masks.extend(
            [
                ("phase_a_succeeds_x_lo_fails", data.success[idx, pa] & ~data.success[idx, lo]),
                ("x_lo_succeeds_phase_a_fails", data.success[idx, lo] & ~data.success[idx, pa]),
                (
                    "both_phase_a_x_lo_succeed_different_iters",
                    data.success[idx, pa]
                    & data.success[idx, lo]
                    & (data.n_iter[idx, pa] != data.n_iter[idx, lo]),
                ),
            ]
        )
    masks.append(("all_baselines_fail", ~np.any(data.success[idx], axis=1)))
    for name, mask in masks:
        if not np.any(mask):
            lines.append(f"  {name}: n=0")
            continue
        sub_idx = idx[mask]
        sub_chosen = chosen[mask]
        m = evaluate_policy(name, sub_chosen, sub_idx, data, best)
        lines.append(f"  {name}: n={len(sub_idx)} fail={m['failure_rate']:.4f} score={m['score']:.4f}")
    if data.state_type is not None:
        for state_type in np.unique(data.state_type[idx]):
            mask = data.state_type[idx] == state_type
            sub_idx = idx[mask]
            sub_chosen = chosen[mask]
            m = evaluate_policy(f"state_type_{int(state_type)}", sub_chosen, sub_idx, data, best)
            lines.append(
                f"  state_type_{int(state_type)}: n={len(sub_idx)} fail={m['failure_rate']:.4f} "
                f"score={m['score']:.4f}"
            )
    for label, arr in (("q", data.q), ("r", data.r), ("s", data.s), ("t_abs", np.abs(data.t))):
        vals = np.asarray(arr)[idx]
        finite = np.isfinite(vals)
        if np.sum(finite) < 4:
            continue
        edges = np.nanpercentile(vals[finite], [0, 25, 50, 75, 100])
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            if hi == edges[-1]:
                mask = (vals >= lo) & (vals <= hi)
            else:
                mask = (vals >= lo) & (vals < hi)
            if not np.any(mask):
                continue
            sub_idx = idx[mask]
            sub_chosen = chosen[mask]
            m = evaluate_policy(f"{label}_bin", sub_chosen, sub_idx, data, best)
            lines.append(
                f"  {label}_bin[{lo:.3e},{hi:.3e}]: n={len(sub_idx)} "
                f"fail={m['failure_rate']:.4f} score={m['score']:.4f}"
            )
    return lines


def _candidate_regret(data: BaselineData, idx: np.ndarray, chosen: np.ndarray, best: np.ndarray) -> dict:
    costs = all_candidate_costs(data, FAILURE_SCORE)[idx]
    row = np.arange(len(idx))
    regret = costs[row, chosen] - costs[row, best[idx]]
    failure_regret = (~data.success[idx][row, chosen]) & data.success[idx][row, best[idx]]
    return {
        "mean_regret": float(np.mean(regret)),
        "p95_regret": float(np.percentile(regret, 95)),
        "failure_regret_rate": float(np.mean(failure_regret)),
    }


def _rare_recall_lines(data: BaselineData, idx: np.ndarray, chosen_by_name: dict[str, np.ndarray], best: np.ndarray) -> list[str]:
    lines = ["Rare-class recall by oracle winner:"]
    for model_name, chosen in chosen_by_name.items():
        lines.append(f"  {model_name}:")
        for c, cname in enumerate(data.names):
            mask = best[idx] == c
            recall = float(np.mean(chosen[mask] == c)) if np.any(mask) else math.nan
            lines.append(f"    oracle_{cname}: n={int(np.sum(mask))} recall={recall:.4f}")
    return lines


def _regret_lines(data: BaselineData, idx: np.ndarray, chosen_by_name: dict[str, np.ndarray], best: np.ndarray) -> list[str]:
    lines = ["Regret diagnostics:"]
    for model_name, chosen in chosen_by_name.items():
        r = _candidate_regret(data, idx, chosen, best)
        lines.append(
            f"  {model_name}: mean={r['mean_regret']:.4f} p95={r['p95_regret']:.4f} "
            f"failure_regret={r['failure_regret_rate']:.4f}"
        )
    return lines


def _predicted_cost_diagnostics(
    title: str,
    data: BaselineData,
    idx: np.ndarray,
    best: np.ndarray,
    pred_cost: np.ndarray,
    psuccess: np.ndarray | None = None,
    iter_pred: np.ndarray | None = None,
) -> list[str]:
    lines = [f"Predicted-cost diagnostics for {title}:"]
    for c, name in enumerate(data.names):
        parts = [
            f"  {name}: mean={np.mean(pred_cost[:, c]):.4f}",
            f"median={np.median(pred_cost[:, c]):.4f}",
            f"p10={np.percentile(pred_cost[:, c], 10):.4f}",
            f"p90={np.percentile(pred_cost[:, c], 90):.4f}",
        ]
        if psuccess is not None:
            parts.append(f"p_success_mean={np.mean(psuccess[:, c]):.4f}")
        if iter_pred is not None:
            parts.append(f"iter_mean={np.mean(iter_pred[:, c]):.4f}")
        lines.append(" ".join(parts))
    lines.append(f"Predicted-cost by oracle winner for {title}:")
    for winner, wname in enumerate(data.names):
        mask = best[idx] == winner
        if not np.any(mask):
            continue
        vals = ", ".join(f"{name}:{np.mean(pred_cost[mask, c]):.4f}" for c, name in enumerate(data.names))
        lines.append(f"  oracle_{wname}: {vals}")
    return lines


def _pairwise_ordering_lines(
    title: str,
    data: BaselineData,
    idx: np.ndarray,
    ranks: np.ndarray,
    scores: np.ndarray,
    higher_is_better: bool,
    best: np.ndarray,
) -> list[str]:
    lines = [f"Pairwise ordering diagnostics for {title}:"]
    for a in range(len(data.names)):
        for b in range(a + 1, len(data.names)):
            pred_a_better = scores[:, a] > scores[:, b] if higher_is_better else scores[:, a] < scores[:, b]
            true_a_better = ranks[idx, a] < ranks[idx, b]
            acc = float(np.mean(pred_a_better == true_a_better))
            lines.append(f"  {data.names[a]} vs {data.names[b]}: acc={acc:.4f}")
            sub = []
            for w, wname in enumerate(data.names):
                mask = best[idx] == w
                if np.any(mask):
                    sub.append(f"oracle_{wname}:{np.mean((pred_a_better == true_a_better)[mask]):.4f}")
            lines.append("    " + " ".join(sub))
    return lines


def write_summary_report(
    path: str | Path,
    data: BaselineData,
    splits: dict[str, np.ndarray],
    best: np.ndarray,
    policy_metrics: list[dict],
    validation_sweeps: list[dict],
    confusions: dict[str, np.ndarray],
    subset_lines: list[str],
    extra_lines: list[str] | None = None,
):
    lines: list[str] = []
    lines.append("Ranking-aware abstaining baseline selector report")
    lines.append("")
    lines.append(f"Reference PhaseA score: {REFERENCE_PHASE_A_SCORE:.2f}")
    lines.append(f"Reference oracle best-baseline score: {REFERENCE_ORACLE_SCORE:.2f}")
    lines.append(f"Reference oracle gap: {REFERENCE_PHASE_A_SCORE - REFERENCE_ORACLE_SCORE:.2f}")
    lines.append("Experiment 11 reference:")
    lines.append(f"  PhaseA score={EXPERIMENT11_PHASE_A_SCORE:.4f} failure={EXPERIMENT11_PHASE_A_FAILURE:.4f}")
    lines.append(f"  oracle score={EXPERIMENT11_ORACLE_SCORE:.4f} failure={EXPERIMENT11_ORACLE_FAILURE:.4f}")
    lines.append(
        f"  full_classifier score={EXPERIMENT11_FULL_CLASSIFIER_SCORE:.4f} "
        f"failure={EXPERIMENT11_FULL_CLASSIFIER_FAILURE:.4f}"
    )
    lines.append(
        f"  old_pair_abstain score={EXPERIMENT11_PAIR_ABSTAIN_SCORE:.4f} "
        f"failure={EXPERIMENT11_PAIR_ABSTAIN_FAILURE:.4f}"
    )
    lines.append("")
    lines.extend(_split_audit(data, best, splits))
    lines.append("")
    lines.append("Validation margin sweep:")
    for m in validation_sweeps:
        lines.append(
            f"  {m['name']:36s} fallback={m.get('fallback',''):8s} margin={m.get('margin',0):4.1f} "
            f"fail={m['failure_rate']:.4f} score={m['score']:.4f} override={m.get('override_rate',0):.4f}"
        )
    lines.append("")
    lines.append("Held-out test policies:")
    for m in policy_metrics:
        lines.append("  " + _fmt_metrics(m))
        if "fallback" in m:
            lines.append(
                f"    fallback={m['fallback']} margin={m['margin']:.2f} overrides={m['n_override']} "
                f"override_rate={m['override_rate']:.4f} win={m['override_win_rate']:.4f} "
                f"loss={m['override_loss_rate']:.4f} gain={m['avg_gain_when_win']:.4f} "
                f"loss_amt={m['avg_loss_when_loss']:.4f}"
            )
        gap = REFERENCE_PHASE_A_SCORE - REFERENCE_ORACLE_SCORE
        if gap > 0:
            captured = (REFERENCE_PHASE_A_SCORE - m["score"]) / gap
            lines.append(f"    reference_gap_captured={captured:.4f}")
        lines.append(f"    chosen_frequency={dict(zip(data.names, m['chosen_frequency'], strict=False))}")
    lines.append("")
    lines.append("Confusion matrices (rows oracle best, cols predicted):")
    for name, cm in confusions.items():
        lines.append(f"  {name}:")
        lines.append("    " + " ".join(f"{n:>9s}" for n in data.names))
        for i, row in enumerate(cm):
            lines.append(f"    {data.names[i]:>9s} " + " ".join(f"{int(v):9d}" for v in row))
    lines.append("")
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    lines.extend(subset_lines)
    lines.append("")
    lines.append("Final selection notes:")
    if policy_metrics:
        best_score = min(policy_metrics, key=lambda m: m["score"])
        best_fail = min(policy_metrics, key=lambda m: m["failure_rate"])
        lines.append(f"  Best model by held-out score: {best_score['name']} score={best_score['score']:.4f}")
        lines.append(f"  Best model by failure rate: {best_fail['name']} fail={best_fail['failure_rate']:.4f}")
        deployable = [
            m
            for m in policy_metrics
            if not m["name"].startswith("always_") and m["name"] != "oracle_best_baseline"
        ]
        if deployable:
            best_deployable = min(deployable, key=lambda m: m["score"])
            best_deployable_fail = min(deployable, key=lambda m: m["failure_rate"])
            lines.append(
                f"  Best learned/deployable policy by score: {best_deployable['name']} "
                f"score={best_deployable['score']:.4f}"
            )
            lines.append(
                f"  Best learned/deployable policy by failure rate: {best_deployable_fail['name']} "
                f"fail={best_deployable_fail['failure_rate']:.4f}"
            )
        safe = [
            m
            for m in policy_metrics
            if np.isfinite(m["p95_iter"])
            and np.isfinite(m["p99_iter"])
            and m["p95_iter"] <= 15.0
            and m["p99_iter"] <= 20.27
        ]
        if safe:
            best_safe = min(safe, key=lambda m: m["score"])
            lines.append(
                f"  Best model not worsening PhaseA p95/p99: {best_safe['name']} "
                f"score={best_safe['score']:.4f}"
            )
        learned_for_criterion = deployable if deployable else policy_metrics
        best_learned = min(learned_for_criterion, key=lambda m: m["score"])
        beats_full = (
            best_learned["score"] < EXPERIMENT11_FULL_CLASSIFIER_SCORE
            and best_learned["failure_rate"] <= EXPERIMENT11_FULL_CLASSIFIER_FAILURE
        )
        lines.append(f"  Beats Experiment-11 full hard classifier criterion: {beats_full}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cost_map_file")
    parser.add_argument("--label_file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidates", default="x_lo,x_hi,midpoint,phase_a")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--fallback_candidates", default="phase_a,x_lo")
    parser.add_argument("--margin_grid", default="0,0.5,1,2,3,5")
    parser.add_argument("--multiclass_margin_grid", default="0,0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--pairwise_compare_margin_grid", default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--multiclass_variants", default="weighted_ce,focal,soft_regret")
    parser.add_argument("--pair_model_variants", default="original,weighted,pairwise_compare")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden_dim_classifier", type=int, default=16)
    parser.add_argument("--hidden_dim_pair", type=int, default=32)
    parser.add_argument("--success_weight", type=float, default=1.0)
    parser.add_argument("--iter_weight", type=float, default=0.05)
    parser.add_argument("--ranking_weight", type=float, default=1.0)
    parser.add_argument("--weighted_rank_lambda", type=float, default=2.0)
    parser.add_argument("--rank_margin", type=float, default=1.0)
    parser.add_argument("--failure_penalty", type=float, default=100.0)
    parser.add_argument("--winner_max_weight", type=float, default=10.0)
    parser.add_argument("--pairwise_compare_hidden_dim", type=int, default=32)
    parser.add_argument("--soft_regret_temperature", default="0.5,1.0,2.0")
    parser.add_argument("--limit_states", type=int, default=None, help="Debug only: use first N states.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    candidate_names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    fallback_names = [x.strip() for x in args.fallback_candidates.split(",") if x.strip()]
    margins = [float(x) for x in args.margin_grid.split(",") if x.strip()]
    multiclass_margins = [float(x) for x in args.multiclass_margin_grid.split(",") if x.strip()]
    pairwise_margins = [float(x) for x in args.pairwise_compare_margin_grid.split(",") if x.strip()]
    multiclass_variants = [x.strip() for x in args.multiclass_variants.split(",") if x.strip()]
    pair_model_variants = [x.strip() for x in args.pair_model_variants.split(",") if x.strip()]
    soft_temperatures = [float(x) for x in args.soft_regret_temperature.split(",") if x.strip()]

    data = load_cost_map_baseline_data(
        args.cost_map_file,
        candidate_names,
        args.label_file,
        limit_states=args.limit_states,
    )
    best, ranks = build_oracle_labels(data)
    train_idx, val_idx, test_idx = build_state_splits(len(data.q), args.split_seed)
    splits = {"train": train_idx, "validation": val_idx, "test": test_idx}

    x_min = make_minimal_classifier_features(data)
    x_full = make_full_classifier_features(data)
    pair_x = make_pair_features(data)
    true_cost = all_candidate_costs(data, args.failure_penalty)

    min_model, min_norm = train_hard_classifier(
        x_min,
        best,
        train_idx,
        val_idx,
        args.hidden_dim_classifier,
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
    )
    full_model, full_norm = train_hard_classifier(
        x_full,
        best,
        train_idx,
        val_idx,
        args.hidden_dim_classifier,
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
    )
    pair_model, pair_norm = train_pair_selector(
        pair_x,
        data.success,
        data.n_iter,
        ranks,
        train_idx,
        val_idx,
        args.hidden_dim_pair,
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
        args.success_weight,
        args.iter_weight,
        args.ranking_weight,
    )

    min_pred_test, _ = predict_classifier(min_model, min_norm, x_min[test_idx])
    full_pred_test, _ = predict_classifier(full_model, full_norm, x_full[test_idx])
    pair_ps_val, pair_iter_val, pair_cost_val = predict_pair_details(pair_model, pair_norm, pair_x[val_idx])
    pair_ps_test, pair_iter_test, pair_cost_test = predict_pair_details(pair_model, pair_norm, pair_x[test_idx])

    policy_metrics: list[dict] = []
    validation_sweeps: list[dict] = []
    confusions: dict[str, np.ndarray] = {}
    chosen_for_subset: np.ndarray | None = None
    best_abstain_score = math.inf
    chosen_by_name: dict[str, np.ndarray] = {}
    extra_lines: list[str] = []
    for i, name in enumerate(data.names):
        policy_metrics.append(evaluate_policy(f"always_{name}", np.full(len(test_idx), i), test_idx, data, best))
    policy_metrics.append(evaluate_policy("oracle_best_baseline", best[test_idx], test_idx, data, best))
    policy_metrics.append(evaluate_policy("hard_classifier_minimal", min_pred_test, test_idx, data, best))
    policy_metrics.append(evaluate_policy("hard_classifier_full", full_pred_test, test_idx, data, best))
    chosen_by_name["hard_classifier_minimal"] = min_pred_test
    chosen_by_name["hard_classifier_full"] = full_pred_test

    raw_pair_selected = np.argmin(pair_cost_test, axis=1)
    policy_metrics.append(evaluate_policy("pair_selector_raw_no_abstain", raw_pair_selected, test_idx, data, best))
    chosen_by_name["pair_selector_raw_no_abstain"] = raw_pair_selected
    confusions.update(
        {
            "hard_classifier_minimal": confusion_matrix(min_pred_test, best[test_idx], len(data.names)),
            "hard_classifier_full": confusion_matrix(full_pred_test, best[test_idx], len(data.names)),
            "pair_selector_raw_no_abstain": confusion_matrix(raw_pair_selected, best[test_idx], len(data.names)),
        }
    )
    extra_lines.extend(_predicted_cost_diagnostics("original_pair_val", data, val_idx, best, pair_cost_val, pair_ps_val, pair_iter_val))
    extra_lines.extend(_pairwise_ordering_lines("original_pair_val", data, val_idx, ranks, pair_cost_val, False, best))

    soft_targets_by_temp = {
        temp: make_soft_regret_targets(true_cost, temp) for temp in soft_temperatures
    }
    multiclass_models: dict[str, tuple[MLP, dict]] = {}
    for variant in multiclass_variants:
        if variant == "soft_regret":
            for temp, targets in soft_targets_by_temp.items():
                name = f"multiclass_soft_regret_T{temp:g}"
                model, norm = train_multiclass_selector(
                    x_full,
                    best,
                    train_idx,
                    val_idx,
                    args.hidden_dim_classifier,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.patience,
                    "soft_regret",
                    soft_targets=targets,
                )
                multiclass_models[name] = (model, norm)
        elif variant in ("weighted_ce", "focal"):
            name = f"multiclass_{variant}"
            model, norm = train_multiclass_selector(
                x_full,
                best,
                train_idx,
                val_idx,
                args.hidden_dim_classifier,
                args.epochs,
                args.batch_size,
                args.lr,
                args.patience,
                "weighted_ce" if variant == "weighted_ce" else "focal",
            )
            multiclass_models[name] = (model, norm)
        else:
            raise ValueError(f"Unknown multiclass variant {variant}")

    for model_name, (model, norm) in multiclass_models.items():
        pred_val, prob_val = predict_classifier(model, norm, x_full[val_idx])
        pred_test, prob_test = predict_classifier(model, norm, x_full[test_idx])
        m_test = evaluate_policy(model_name, pred_test, test_idx, data, best)
        policy_metrics.append(m_test)
        chosen_by_name[model_name] = pred_test
        confusions[model_name] = confusion_matrix(pred_test, best[test_idx], len(data.names))
        if m_test["score"] < best_abstain_score:
            best_abstain_score = m_test["score"]
            chosen_for_subset = pred_test
        for fallback_name in fallback_names:
            if fallback_name not in data.names:
                raise ValueError(f"Unknown fallback {fallback_name}; candidates are {data.names}")
            fallback_id = data.names.index(fallback_name)
            val_candidates = []
            for margin in multiclass_margins:
                m_val, _ = evaluate_score_abstaining_policy(
                    f"{model_name}_abstain_val_{fallback_name}_m{margin:g}",
                    prob_val,
                    fallback_id,
                    margin,
                    val_idx,
                    data,
                    best,
                    ranks,
                    True,
                )
                validation_sweeps.append(m_val)
                val_candidates.append((m_val["score"], margin))
            _, chosen_margin = min(val_candidates, key=lambda x: x[0])
            m_test, chosen = evaluate_score_abstaining_policy(
                f"{model_name}_abstain_{fallback_name}_best_val_margin",
                prob_test,
                fallback_id,
                chosen_margin,
                test_idx,
                data,
                best,
                ranks,
                True,
            )
            policy_metrics.append(m_test)
            chosen_by_name[m_test["name"]] = chosen
            if m_test["score"] < best_abstain_score:
                best_abstain_score = m_test["score"]
                chosen_for_subset = chosen

    if raw_pair_selected is not None and best_abstain_score == math.inf:
        chosen_for_subset = raw_pair_selected
        best_abstain_score = evaluate_policy("tmp", raw_pair_selected, test_idx, data, best)["score"]

    for fallback_name in fallback_names:
        if fallback_name not in data.names:
            raise ValueError(f"Unknown fallback {fallback_name}; candidates are {data.names}")
        fallback_id = data.names.index(fallback_name)
        val_candidates = []
        for margin in margins:
            m_val, _ = evaluate_abstaining_policy(
                f"pair_abstain_val_{fallback_name}_m{margin:g}",
                pair_cost_val,
                fallback_id,
                margin,
                val_idx,
                data,
                best,
                ranks,
            )
            validation_sweeps.append(m_val)
            val_candidates.append((m_val["score"], margin))
        _, chosen_margin = min(val_candidates, key=lambda x: x[0])
        m_test, chosen = evaluate_abstaining_policy(
            f"pair_abstain_{fallback_name}_best_val_margin",
            pair_cost_test,
            fallback_id,
            chosen_margin,
            test_idx,
            data,
            best,
            ranks,
        )
        policy_metrics.append(m_test)
        chosen_by_name[m_test["name"]] = chosen
        if m_test["score"] < best_abstain_score:
            best_abstain_score = m_test["score"]
            chosen_for_subset = chosen

    weighted_pair_model = None
    weighted_pair_norm = None
    if "weighted" in pair_model_variants:
        weighted_pair_model, weighted_pair_norm = train_weighted_pair_selector(
            pair_x,
            data.success,
            data.n_iter,
            ranks,
            best,
            true_cost,
            train_idx,
            val_idx,
            args.hidden_dim_pair,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
            args.success_weight,
            args.iter_weight,
            args.weighted_rank_lambda,
            args.rank_margin,
            args.failure_penalty,
            args.winner_max_weight,
        )
        wps_val, wit_val, wcost_val = predict_pair_details(weighted_pair_model, weighted_pair_norm, pair_x[val_idx])
        wps_test, wit_test, wcost_test = predict_pair_details(weighted_pair_model, weighted_pair_norm, pair_x[test_idx])
        weighted_raw = np.argmin(wcost_test, axis=1)
        m_test = evaluate_policy("weighted_pair_raw_no_abstain", weighted_raw, test_idx, data, best)
        policy_metrics.append(m_test)
        chosen_by_name["weighted_pair_raw_no_abstain"] = weighted_raw
        confusions["weighted_pair_raw_no_abstain"] = confusion_matrix(weighted_raw, best[test_idx], len(data.names))
        extra_lines.extend(_predicted_cost_diagnostics("weighted_pair_val", data, val_idx, best, wcost_val, wps_val, wit_val))
        extra_lines.extend(_pairwise_ordering_lines("weighted_pair_val", data, val_idx, ranks, wcost_val, False, best))
        if m_test["score"] < best_abstain_score:
            best_abstain_score = m_test["score"]
            chosen_for_subset = weighted_raw
        for fallback_name in fallback_names:
            fallback_id = data.names.index(fallback_name)
            val_candidates = []
            for margin in margins:
                m_val, _ = evaluate_abstaining_policy(
                    f"weighted_pair_abstain_val_{fallback_name}_m{margin:g}",
                    wcost_val,
                    fallback_id,
                    margin,
                    val_idx,
                    data,
                    best,
                    ranks,
                )
                validation_sweeps.append(m_val)
                val_candidates.append((m_val["score"], margin))
            _, chosen_margin = min(val_candidates, key=lambda x: x[0])
            m_test, chosen = evaluate_abstaining_policy(
                f"weighted_pair_abstain_{fallback_name}_best_val_margin",
                wcost_test,
                fallback_id,
                chosen_margin,
                test_idx,
                data,
                best,
                ranks,
            )
            policy_metrics.append(m_test)
            chosen_by_name[m_test["name"]] = chosen
            if m_test["score"] < best_abstain_score:
                best_abstain_score = m_test["score"]
                chosen_for_subset = chosen

    cmp_model = None
    cmp_norm = None
    cmp_pairs = None
    if "pairwise_compare" in pair_model_variants:
        cmp_x, cmp_pairs = make_pairwise_comparison_features(pair_x)
        cmp_model, cmp_norm = train_pairwise_comparator(
            cmp_x,
            cmp_pairs,
            ranks,
            best,
            true_cost,
            data.success,
            train_idx,
            val_idx,
            args.pairwise_compare_hidden_dim,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
            args.winner_max_weight,
        )
        borda_val, copeland_val = predict_pairwise_scores(cmp_model, cmp_norm, cmp_x[val_idx], len(data.names))
        borda_test, copeland_test = predict_pairwise_scores(cmp_model, cmp_norm, cmp_x[test_idx], len(data.names))
        for score_name, val_scores, test_scores in (
            ("pairwise_borda", borda_val, borda_test),
            ("pairwise_copeland", copeland_val, copeland_test),
        ):
            raw = np.argmax(test_scores, axis=1)
            m_test = evaluate_policy(f"{score_name}_raw_no_abstain", raw, test_idx, data, best)
            policy_metrics.append(m_test)
            chosen_by_name[m_test["name"]] = raw
            confusions[m_test["name"]] = confusion_matrix(raw, best[test_idx], len(data.names))
            extra_lines.extend(_pairwise_ordering_lines(f"{score_name}_val", data, val_idx, ranks, val_scores, True, best))
            if m_test["score"] < best_abstain_score:
                best_abstain_score = m_test["score"]
                chosen_for_subset = raw
            for fallback_name in fallback_names:
                fallback_id = data.names.index(fallback_name)
                val_candidates = []
                for margin in pairwise_margins:
                    m_val, _ = evaluate_score_abstaining_policy(
                        f"{score_name}_abstain_val_{fallback_name}_m{margin:g}",
                        val_scores,
                        fallback_id,
                        margin,
                        val_idx,
                        data,
                        best,
                        ranks,
                        True,
                    )
                    validation_sweeps.append(m_val)
                    val_candidates.append((m_val["score"], margin))
                _, chosen_margin = min(val_candidates, key=lambda x: x[0])
                m_test, chosen = evaluate_score_abstaining_policy(
                    f"{score_name}_abstain_{fallback_name}_best_val_margin",
                    test_scores,
                    fallback_id,
                    chosen_margin,
                    test_idx,
                    data,
                    best,
                    ranks,
                    True,
                )
                policy_metrics.append(m_test)
                chosen_by_name[m_test["name"]] = chosen
                if m_test["score"] < best_abstain_score:
                    best_abstain_score = m_test["score"]
                    chosen_for_subset = chosen

    if chosen_for_subset is None:
        chosen_for_subset = raw_pair_selected
    extra_lines.extend(_rare_recall_lines(data, test_idx, chosen_by_name, best))
    extra_lines.extend(_regret_lines(data, test_idx, chosen_by_name, best))
    subset_lines = _subset_diagnostics(data, test_idx, chosen_for_subset, best, "best_test_selector_policy")
    write_summary_report(
        args.report,
        data,
        splits,
        best,
        policy_metrics,
        validation_sweeps,
        confusions,
        subset_lines,
        extra_lines,
    )

    artifact = {
        "candidate_names": data.names,
        "split_seed": args.split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "best_labels": best,
        "minimal_classifier": {
            "state_dict": min_model.state_dict(),
            "normalizer": min_norm,
            "hidden_dim": args.hidden_dim_classifier,
        },
        "full_classifier": {
            "state_dict": full_model.state_dict(),
            "normalizer": full_norm,
            "hidden_dim": args.hidden_dim_classifier,
        },
        "pair_selector": {
            "state_dict": pair_model.state_dict(),
            "normalizer": pair_norm,
            "hidden_dim": args.hidden_dim_pair,
            "success_weight": args.success_weight,
            "iter_weight": args.iter_weight,
            "ranking_weight": args.ranking_weight,
        },
        "multiclass_selectors": {
            name: {
                "state_dict": model.state_dict(),
                "normalizer": norm,
                "hidden_dim": args.hidden_dim_classifier,
            }
            for name, (model, norm) in multiclass_models.items()
        },
        "weighted_pair_selector": {
            "state_dict": weighted_pair_model.state_dict() if weighted_pair_model is not None else None,
            "normalizer": weighted_pair_norm,
            "hidden_dim": args.hidden_dim_pair,
        },
        "pairwise_comparator": {
            "state_dict": cmp_model.state_dict() if cmp_model is not None else None,
            "normalizer": cmp_norm,
            "pairs": cmp_pairs,
            "hidden_dim": args.pairwise_compare_hidden_dim,
        },
        "policy_metrics": policy_metrics,
        "validation_sweeps": validation_sweeps,
    }
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Wrote selector artifact to {args.output}")
    print(f"Wrote selector report to {args.report}")


if __name__ == "__main__":
    main()
