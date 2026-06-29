from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ._nn_common import (
    FeatureTransformStats,
    TinyMLP_Logit,
    apply_feature_transform,
    apply_robust_minmax,
    compute_x_bounds_from_inputs,
)
from ._nn_dataset import read_solver_label_dataset, read_training_dataset
from ._nn_hdf5 import (
    append_nn_to_eos_file,
    build_eos_metadata,
    eos_nn_metadata,
    install_nn_model,
    write_nn_hdf5,
)
from ._nn_infer import (
    OUT_KIND_LINEAR,
    OUT_KIND_LOG_LINEAR,
    OUT_KIND_X_BOUNDED,
    save_inference_bundle,
)
from .nn_c2p.nn_c2p_generate_dataset import generate_dataset

EXPORT_USE_INCLUDE_GUARD: bool = True
EXPORT_ADD_AUDIT_COMMENTS: bool = True


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        print("Using Apple Silicon device")
        return torch.device("mps")
    print("Using CPU (no supported device found)")
    return torch.device("cpu")


def load_dataset_from_numpy(
    data_np,
    *,
    n_out: int = 1,
    drop_nonfinite_rows: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(data_np, dtype=torch.float32)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (N,n). Got shape {tuple(data.shape)}")
    if n_out < 1:
        raise ValueError("n_out must be >= 1.")
    if data.shape[1] <= n_out:
        raise ValueError(
            f"data must have > n_out columns (features + targets). "
            f"Got n={data.shape[1]}, n_out={n_out}"
        )
    if data.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 rows to split train/val. Got N={data.shape[0]}"
        )

    finite_mask = torch.isfinite(data).all(dim=1)
    if not finite_mask.all():
        bad = int((~finite_mask).sum().item())
        if drop_nonfinite_rows:
            data = data[finite_mask]
            if data.shape[0] < 2:
                raise ValueError(
                    f"After dropping {bad} non-finite rows, N={data.shape[0]} < 2."
                )
        else:
            raise ValueError(
                f"Dataset contains {bad} rows with NaN/Inf. "
                f"Set drop_nonfinite_rows=True to drop them."
            )

    return data[:, :-n_out].contiguous(), data[:, -n_out:].contiguous()


@torch.no_grad()
def filter_invalid_width_rows(
    X_raw: torch.Tensor,
    y_x: torch.Tensor,
    *,
    q_idx: int,
    width_tiny: float = 1.0e-12,
    drop_invalid: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    q = X_raw[:, q_idx : q_idx + 1]
    width = 1.0 + q
    ok = torch.isfinite(width).all(dim=1) & (width.squeeze(1) > float(width_tiny))
    n_bad = int((~ok).sum().item())
    if n_bad == 0:
        return X_raw, y_x, 0

    if not drop_invalid:
        qb = q[~ok][:5].reshape(-1).tolist()
        raise ValueError(
            f"Found {n_bad} rows with invalid Palenzuela width (1+q) <= {width_tiny} or non-finite. "
            f"Example bad q values (up to 5): {qb}. Refusing to proceed."
        )

    X2 = X_raw[ok]
    y2 = y_x[ok]
    if X2.shape[0] < 2:
        raise ValueError(
            f"After dropping {n_bad} invalid-width rows, N={X2.shape[0]} < 2."
        )
    print(
        f"Warning: dropped {n_bad} rows with invalid Palenzuela width (1+q) <= {width_tiny} or non-finite."
    )
    return X2, y2, n_bad


@torch.no_grad()
def clamp_x_to_bounds(
    X_raw: torch.Tensor,
    x: torch.Tensor,
    *,
    q_idx: int,
    s_idx: int,
    y_eps: float,
) -> Tuple[torch.Tensor, float]:
    x_lo, x_hi, width = compute_x_bounds_from_inputs(X_raw, q_idx=q_idx, s_idx=s_idx)
    w = torch.clamp(width.to(torch.float32), min=1.0e-12)
    lo = x_lo.to(torch.float32) + float(y_eps) * w
    hi = x_hi.to(torch.float32) - float(y_eps) * w

    x32 = x.to(torch.float32)
    x_clamped = torch.clamp(x32, min=lo, max=hi)
    changed = (x_clamped != x32).any(dim=1).float().mean().item()
    return x_clamped, float(changed)


@torch.no_grad()
def x_to_y01_target(
    X_raw: torch.Tensor,
    x: torch.Tensor,
    *,
    q_idx: int,
    s_idx: int,
    y_eps: float,
) -> torch.Tensor:
    x_lo, _, width = compute_x_bounds_from_inputs(X_raw, q_idx=q_idx, s_idx=s_idx)
    w = torch.clamp(width.to(torch.float32), min=1.0e-12)
    x32 = x.to(torch.float32)
    y01 = (x32 - x_lo.to(torch.float32)) / w
    return torch.clamp(y01, min=float(y_eps), max=1.0 - float(y_eps))


@torch.no_grad()
def fit_output_linear_scaling(
    ytr: torch.Tensor,
    *,
    q_lo: float,
    q_hi: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    return fit_robust_minmax(ytr, q_lo=q_lo, q_hi=q_hi, eps=eps)


@torch.no_grad()
def linear_to_y01_target(
    y: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    *,
    y_eps: float,
) -> torch.Tensor:
    lo = stats["lo"].to(device=y.device, dtype=torch.float32)
    hi = stats["hi"].to(device=y.device, dtype=torch.float32)
    invrng = stats["invrng"].to(device=y.device, dtype=torch.float32)
    y01 = (torch.clamp(y.to(torch.float32), min=lo, max=hi) - lo) * invrng
    return torch.clamp(y01, min=float(y_eps), max=1.0 - float(y_eps))


def y01_to_linear(
    y01: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    *,
    y_eps: float,
) -> torch.Tensor:
    lo = stats["lo"].to(device=y01.device, dtype=torch.float32)
    invrng = stats["invrng"].to(device=y01.device, dtype=torch.float32)
    y01c = torch.clamp(y01.to(torch.float32), min=float(y_eps), max=1.0 - float(y_eps))
    return lo + y01c / invrng


@torch.no_grad()
def transform_W_target(W: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.log(torch.clamp(W.to(torch.float32), min=float(eps)))


def inverse_transform_W_target(logW: torch.Tensor) -> torch.Tensor:
    return torch.exp(logW.to(torch.float32))


@torch.no_grad()
def transform_positive_target(x: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x.to(torch.float32), min=float(eps)))


def inverse_transform_positive_target(logx: torch.Tensor) -> torch.Tensor:
    return torch.exp(logx.to(torch.float32))


@torch.no_grad()
def _quantile_1d(x: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(x, q)


@torch.no_grad()
def fit_feature_transform(
    Xtr_raw: torch.Tensor,
    *,
    pos_log10_ratio_thresh: float,
    eps: float,
    neg_frac_tol: float = 1e-4,
    min_pos_count: int = 256,
    force_kind: Optional[Dict[int, int]] = None,
    verbose: bool = True,
) -> FeatureTransformStats:
    X = Xtr_raw.detach()
    if X.device.type != "cpu":
        X = X.cpu()

    d = X.shape[1]
    kind = torch.zeros(d, dtype=torch.int32)
    eps32 = torch.tensor(float(eps), dtype=torch.float32)
    decisions = []
    for j in range(d):
        col32 = X[:, j].to(torch.float32)
        if not torch.isfinite(col32).all():
            decisions.append((j, 0, float("nan"), 0, float("nan"), "nonfinite"))
            continue
        cmax = float(col32.max().item())
        if not math.isfinite(cmax) or cmax <= 0.0:
            decisions.append((j, 0, 0.0, 0, 0.0, "no_positive_support"))
            continue
        frac_neg = float((col32 < 0.0).float().mean().item())
        if frac_neg > float(neg_frac_tol):
            decisions.append((j, 0, frac_neg, 0, 0.0, "signed_or_contaminated"))
            continue
        pos32 = col32[col32 > 0.0]
        pos_n = int(pos32.numel())
        if pos_n < int(min_pos_count):
            decisions.append((j, 0, frac_neg, pos_n, 0.0, "too_few_positives"))
            continue
        q01 = torch.clamp(_quantile_1d(pos32, 0.01), min=eps32)
        q99 = torch.clamp(_quantile_1d(pos32, 0.99), min=eps32)
        if q01 <= (100.0 * eps32):
            decisions.append(
                (j, 0, frac_neg, pos_n, float((q99 / q01).item()), "q01_near_eps")
            )
            continue
        ratio = float((q99 / q01).item())
        if ratio > float(pos_log10_ratio_thresh):
            kind[j] = 1
            decisions.append((j, 1, frac_neg, pos_n, ratio, "ratio_ok"))
        else:
            decisions.append((j, 0, frac_neg, pos_n, ratio, "ratio_small"))

    if force_kind is not None:
        for idx, k in force_kind.items():
            if not (0 <= int(idx) < d):
                raise ValueError(f"force_kind index {idx} out of range for d_in={d}")
            kind[int(idx)] = int(k)
        if verbose:
            print(f"Note: applied force_kind overrides: {force_kind}")

    if verbose:
        print("Feature transform decisions (idx kind frac_neg pos_count ratio note):")
        for j, k, fn, pn, r, note in decisions:
            r_s = f"{r:.3e}" if math.isfinite(r) else "nan"
            fn_s = f"{fn:.3e}" if math.isfinite(fn) else "nan"
            print(f"  {j:2d}  {k:1d}  {fn_s:>10s}  {pn:8d}  {r_s:>10s}  {note}")

    return FeatureTransformStats(kind=kind, eps=float(eps))


@torch.no_grad()
def fit_robust_minmax(
    Xtr_t: torch.Tensor,
    *,
    q_lo: float,
    q_hi: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    X = Xtr_t.detach()
    if X.device.type != "cpu":
        X = X.cpu()
    X = X.to(torch.float32)

    d = X.shape[1]
    lo = torch.empty(d, dtype=torch.float32)
    hi = torch.empty(d, dtype=torch.float32)
    for j in range(d):
        col32 = X[:, j].to(torch.float32)
        lo[j] = torch.quantile(col32, q_lo)
        hi[j] = torch.quantile(col32, q_hi)

    rng = hi - lo
    rng = torch.where(rng < float(eps), torch.ones_like(rng), rng)
    return {
        "lo": lo.to(torch.float32),
        "hi": hi.to(torch.float32),
        "invrng": (1.0 / rng).to(torch.float32),
    }


@torch.no_grad()
def fit_standardizer(x: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return {
        "mean": mean.to(torch.float32),
        "std": std.to(torch.float32),
        "invstd": (1.0 / std).to(torch.float32),
    }


@torch.no_grad()
def x_metrics(pred_x: torch.Tensor, true_x: torch.Tensor) -> Tuple[float, float]:
    err = pred_x - true_x
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err * err).mean()).item()
    return mae, rmse


@torch.no_grad()
def x_relative_metrics(
    pred_x: torch.Tensor,
    true_x: torch.Tensor,
    *,
    denom_eps: float,
) -> Tuple[float, float, float]:
    denom = torch.clamp(true_x.abs(), min=float(denom_eps))
    rel = (pred_x - true_x) / denom
    rel_mse = (rel * rel).mean().item()
    rel_mae = rel.abs().mean().item()
    rel_rmse = torch.sqrt((rel * rel).mean()).item()
    return float(rel_mse), float(rel_mae), float(rel_rmse)


def y01_to_x_differentiable_from_qs(
    qs_raw: torch.Tensor,
    y01: torch.Tensor,
    *,
    y_eps: float,
) -> torch.Tensor:
    if qs_raw.ndim != 2 or qs_raw.shape[1] != 2:
        raise ValueError(f"qs_raw must have shape (N,2). Got {tuple(qs_raw.shape)}")
    q = qs_raw[:, 0:1].to(torch.float32)
    s = qs_raw[:, 1:2].to(torch.float32)
    x_lo = 1.0 + q - s
    w = torch.clamp(1.0 + q, min=1.0e-12)
    y01c = torch.clamp(y01.to(torch.float32), min=float(y_eps), max=1.0 - float(y_eps))
    return (x_lo + y01c * w).to(torch.float32)


def relative_error_loss_x(
    pred_x: torch.Tensor,
    true_x: torch.Tensor,
    *,
    denom_eps: float,
    mode: str = "mse",
) -> torch.Tensor:
    denom = torch.clamp(true_x.abs(), min=float(denom_eps)).to(torch.float32)
    rel = (pred_x.to(torch.float32) - true_x.to(torch.float32)) / denom
    if mode == "mse":
        return (rel * rel).mean()
    if mode == "mae":
        return rel.abs().mean()
    if mode == "log1p_mse":
        z = torch.log1p(rel.abs())
        return (z * z).mean()
    raise ValueError(f"Unknown relative loss mode '{mode}'. Use 'mse', 'mae', or 'log1p_mse'.")


def region_distance_loss(
    y_pred: torch.Tensor,
    y_grid: torch.Tensor,
    robust_mask: torch.Tensor,
    fast_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    fast_fallback_weight: float = 0.25,
) -> torch.Tensor:
    y_expanded = y_pred.to(torch.float32)
    if y_expanded.ndim != 2 or y_expanded.shape[1] != 1:
        raise ValueError(f"Expected y_pred shape (N,1). Got {tuple(y_expanded.shape)}")
    grid = y_grid.to(device=y_pred.device, dtype=torch.float32).view(1, -1)
    dist2 = (y_expanded - grid) ** 2

    huge = torch.full_like(dist2, 1.0e6)
    robust_any = robust_mask.any(dim=1, keepdim=True)
    fast_any = fast_mask.any(dim=1, keepdim=True)
    robust_best = torch.where(robust_mask, dist2, huge).min(dim=1).values
    fast_best = torch.where(fast_mask, dist2, huge).min(dim=1).values
    fallback = torch.where(
        robust_any.squeeze(1),
        robust_best,
        torch.where(
            fast_any.squeeze(1),
            fast_fallback_weight * fast_best,
            torch.zeros_like(fast_best),
        ),
    )
    weights = sample_weight.to(device=y_pred.device, dtype=torch.float32).reshape(-1)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (fallback * weights).sum() / denom


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_prob: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    logp = torch.log_softmax(logits.to(torch.float32), dim=1)
    ce = -(target_prob.to(torch.float32) * logp).sum(dim=1)
    weights = sample_weight.to(device=logits.device, dtype=torch.float32).reshape(-1)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (ce * weights).sum() / denom


def y01_to_region_hit_rate(
    y_pred: torch.Tensor,
    y_grid: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    pred = y_pred.detach().cpu().reshape(-1)
    grid = y_grid.detach().cpu().reshape(-1)
    mask_cpu = mask.detach().cpu()
    hit = 0
    total = 0
    for i in range(mask_cpu.shape[0]):
        valid = mask_cpu[i]
        if not bool(valid.any().item()):
            continue
        total += 1
        idx = int(torch.argmin(torch.abs(grid - pred[i])).item())
        if bool(valid[idx].item()):
            hit += 1
    return float(hit) / float(total) if total else 0.0


def _finalize_stats_and_model(
    *,
    model: nn.Module,
    ft_stats: FeatureTransformStats,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
    q_idx: int,
    s_idx: int,
    return_model_on_cpu: bool,
):
    ft_stats = FeatureTransformStats(
        kind=ft_stats.kind.detach().cpu().to(torch.int32),
        eps=float(ft_stats.eps),
        q_idx=int(q_idx),
        s_idx=int(s_idx),
    )
    x_stats = {k: v.detach().cpu().to(torch.float32) for k, v in x_stats.items()}
    y_stats = {k: v.detach().cpu().to(torch.float32) for k, v in y_stats.items()}
    if return_model_on_cpu:
        model = model.to(device="cpu", dtype=torch.float32)
    return model, ft_stats, x_stats, y_stats


def _optimizer_state_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"state": {}, "param_groups": state.get("param_groups", [])}
    for k, v in state.get("state", {}).items():
        if isinstance(v, dict):
            vv: Dict[str, Any] = {}
            for kk, vv0 in v.items():
                vv[kk] = vv0.detach().cpu() if torch.is_tensor(vv0) else vv0
            out["state"][k] = vv
        else:
            out["state"][k] = v
    return out


def _get_rng_snapshot(device: torch.device, gen: torch.Generator) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "torch_rng_state": torch.get_rng_state(),
        "generator_state": gen.get_state(),
        "python_random_state": random.getstate(),
    }
    if device.type == "cuda":
        try:
            snap["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return snap


def _get_backend_flags_snapshot() -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    try:
        flags["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    try:
        flags["float32_matmul_precision"] = str(torch.get_float32_matmul_precision())
    except Exception:
        pass
    return flags


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "fcs.0.weight" in state_dict:
        return state_dict
    for prefix in ("_orig_mod.", "module."):
        key = prefix + "fcs.0.weight"
        if key in state_dict:
            return {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
    return state_dict


def _load_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    strict: bool,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path!r} is missing 'model_state_dict'.")
    model.load_state_dict(_normalize_state_dict_keys(model_state), strict=strict)

    if "opt_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    best_state = ckpt.get("best_model_state_dict")
    if isinstance(best_state, dict):
        best_state = _normalize_state_dict_keys(best_state)

    return {
        "last_epoch": int(ckpt.get("epoch", 0)),
        "best_epoch": int(ckpt.get("best_epoch", 0)),
        "best_rel_rmse": float(
            ckpt.get("best_rel_rmse_combined", ckpt.get("best_rel_rmse_x", float("inf")))
        ),
        "best_state": best_state,
        "rng": ckpt.get("rng"),
        "gen_ep_state": ckpt.get("gen_ep_state"),
    }


def _checkpoint_epoch_from_name(path: str) -> int:
    m = re.search(r"_ep(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _resolve_resume_checkpoint(resume_checkpoint: Optional[str]) -> Optional[str]:
    if resume_checkpoint is None:
        return None
    if os.path.isfile(resume_checkpoint):
        return resume_checkpoint
    if os.path.isdir(resume_checkpoint):
        candidates = [
            os.path.join(resume_checkpoint, name)
            for name in os.listdir(resume_checkpoint)
            if name.endswith(".pt") and os.path.isfile(os.path.join(resume_checkpoint, name))
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No .pt checkpoint files found in directory {resume_checkpoint!r}."
            )
        resolved = max(
            candidates,
            key=lambda p: (_checkpoint_epoch_from_name(p), os.path.getmtime(p)),
        )
        print(f"Resolved resume checkpoint directory {resume_checkpoint!r} -> {resolved!r}")
        return resolved
    raise FileNotFoundError(
        f"resume_checkpoint path {resume_checkpoint!r} does not exist as a file or directory."
    )


def train_regressor(
    data_np,
    *,
    n_out: Optional[int] = None,
    target_mode: str = "x_correction",
    hidden_dim: int = 3,
    n_hidden: int = 3,
    lr: float = 5e-4,
    epochs: int = 2000,
    batch_size: int = 4096,
    val_frac: float = 0.2,
    seed: int = 0,
    deterministic: bool = False,
    patience: int = 250,
    min_delta: float = 0.0,
    weight_decay: float = 1e-6,
    grad_clip_norm: Optional[float] = 1.0,
    drop_nonfinite_rows: bool = True,
    return_model_on_cpu: bool = True,
    log_path: str = "training_log.txt",
    pos_log10_ratio_thresh: float = 1e3,
    robust_mm_qlo: float = 0.001,
    robust_mm_qhi: float = 0.999,
    neg_frac_tol: float = 1e-4,
    min_pos_count: int = 256,
    force_kind: Optional[Dict[int, int]] = None,
    y_eps: float = 1e-7,
    q_idx: int = 0,
    s_idx: int = 2,
    width_tiny: float = 1.0e-12,
    drop_invalid_width_rows: bool = True,
    rel_denom_eps: float = 1.0e-12,
    rel_loss_mode: str = "mse",
    checkpoint_every: int = 200,
    checkpoint_dir: str = "checkpoints",
    checkpoint_export_header: bool = True,
    checkpoint_prefix: str = "tiny_mlp",
    resume_checkpoint: Optional[str] = None,
    resume_strict: bool = True,
    verbose_transforms: bool = True,
):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    random.seed(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    device = pick_device()

    gen = torch.Generator().manual_seed(seed)
    if n_out is None:
        n_out = 1 if target_mode in ("x_correction", "x_best_correction") else 3

    X_raw, y = load_dataset_from_numpy(
        data_np, n_out=n_out, drop_nonfinite_rows=drop_nonfinite_rows
    )
    if X_raw.shape[1] <= max(q_idx, s_idx):
        raise ValueError(
            f"Input has d_in={X_raw.shape[1]} but requires q_idx={q_idx}, s_idx={s_idx}."
        )
    if target_mode not in ("x_correction", "x_best_correction", "rho_T_W"):
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}. "
            "Expected 'x_correction', 'x_best_correction', or 'rho_T_W'."
        )
    if target_mode == "rho_T_W":
        if y.shape[1] != 3:
            raise ValueError(
                f"Expected targets (rho, T, W) with d_out=3. Got d_out={y.shape[1]}"
            )
        if not bool((y > 0.0).all().item()):
            raise ValueError("Targets (rho, T, W) must all be strictly positive.")
    else:
        if y.shape[1] != 1:
            raise ValueError(
                f"Expected target (x_exact) with d_out=1. Got d_out={y.shape[1]}"
            )
        X_raw, y, n_bad_width = filter_invalid_width_rows(
            X_raw,
            y,
            q_idx=q_idx,
            width_tiny=width_tiny,
            drop_invalid=drop_invalid_width_rows,
        )
        y, changed_frac = clamp_x_to_bounds(
            X_raw,
            y,
            q_idx=q_idx,
            s_idx=s_idx,
            y_eps=y_eps,
        )
        print(
            f"Prepared bounded-x targets: dropped_invalid_width_rows={n_bad_width} "
            f"clamped_target_frac={changed_frac:.6e}"
        )

    N = X_raw.shape[0]
    n_val = max(1, min(int(val_frac * N), N - 1))
    perm = torch.randperm(N, generator=gen)
    perm_hash = hashlib.sha1(perm.cpu().numpy().tobytes()).hexdigest()[:16]
    print(f"Split: N={N} n_val={n_val} seed={seed} perm_sha1_16={perm_hash}")

    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xtr_raw, ytr = X_raw[tr_idx], y[tr_idx]
    Xva_raw, yva = X_raw[val_idx], y[val_idx]

    if target_mode == "rho_T_W":
        ytr_log = torch.cat(
            (
                transform_positive_target(ytr[:, 0:1]),
                transform_positive_target(ytr[:, 1:2]),
                transform_positive_target(ytr[:, 2:3]),
            ),
            dim=1,
        )
        yva_log = torch.cat(
            (
                transform_positive_target(yva[:, 0:1]),
                transform_positive_target(yva[:, 1:2]),
                transform_positive_target(yva[:, 2:3]),
            ),
            dim=1,
        )
        y_out_stats = fit_output_linear_scaling(
            ytr_log, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi
        )
        ytr_y01 = linear_to_y01_target(ytr_log, y_out_stats, y_eps=y_eps)
        yva_y01 = linear_to_y01_target(yva_log, y_out_stats, y_eps=y_eps)
    else:
        ytr_y01 = x_to_y01_target(
            Xtr_raw, ytr, q_idx=q_idx, s_idx=s_idx, y_eps=y_eps
        )
        yva_y01 = x_to_y01_target(
            Xva_raw, yva, q_idx=q_idx, s_idx=s_idx, y_eps=y_eps
        )
        y_out_stats = {
            "lo": torch.zeros(1, dtype=torch.float32),
            "hi": torch.ones(1, dtype=torch.float32),
            "invrng": torch.ones(1, dtype=torch.float32),
        }

    ft_stats = fit_feature_transform(
        Xtr_raw,
        pos_log10_ratio_thresh=pos_log10_ratio_thresh,
        eps=1e-30,
        neg_frac_tol=neg_frac_tol,
        min_pos_count=min_pos_count,
        force_kind=force_kind,
        verbose=verbose_transforms,
    )
    ft_stats.q_idx = int(q_idx)
    ft_stats.s_idx = int(s_idx)

    Xtr_t = apply_feature_transform(Xtr_raw, ft_stats)
    Xva_t = apply_feature_transform(Xva_raw, ft_stats)
    mm_stats = fit_robust_minmax(Xtr_t, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi)
    Xtr01, tr_clip_frac = apply_robust_minmax(Xtr_t, mm_stats)
    Xva01, va_clip_frac = apply_robust_minmax(Xva_t, mm_stats)

    x_std_stats = fit_standardizer(Xtr_t)
    y_std_stats = fit_standardizer(ytr)
    x_stats: Dict[str, torch.Tensor] = {
        "lo": mm_stats["lo"].detach().cpu().to(torch.float32),
        "hi": mm_stats["hi"].detach().cpu().to(torch.float32),
        "invrng": mm_stats["invrng"].detach().cpu().to(torch.float32),
        "mean": x_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": x_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": x_std_stats["invstd"].detach().cpu().to(torch.float32),
    }
    if target_mode == "rho_T_W":
        out_kind = torch.tensor(
            [OUT_KIND_LOG_LINEAR, OUT_KIND_LOG_LINEAR, OUT_KIND_LOG_LINEAR],
            dtype=torch.int32,
        )
    else:
        out_kind = torch.tensor([OUT_KIND_X_BOUNDED], dtype=torch.int32)

    dropped_invalid_width_value = float(n_bad_width) if target_mode == "x_correction" else 0.0
    y_stats: Dict[str, torch.Tensor] = {
        "mean": y_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": y_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": y_std_stats["invstd"].detach().cpu().to(torch.float32),
        "out_kind": out_kind,
        "out_lo": y_out_stats["lo"].detach().cpu().to(torch.float32),
        "out_hi": y_out_stats["hi"].detach().cpu().to(torch.float32),
        "out_invrng": y_out_stats["invrng"].detach().cpu().to(torch.float32),
        "y_eps": torch.tensor(float(y_eps), dtype=torch.float32),
        "robust_mm_qlo": torch.tensor(float(robust_mm_qlo), dtype=torch.float32),
        "robust_mm_qhi": torch.tensor(float(robust_mm_qhi), dtype=torch.float32),
        "pos_log10_ratio_thresh": torch.tensor(float(pos_log10_ratio_thresh), dtype=torch.float32),
        "neg_frac_tol": torch.tensor(float(neg_frac_tol), dtype=torch.float32),
        "min_pos_count": torch.tensor(float(min_pos_count), dtype=torch.float32),
        "width_tiny": torch.tensor(float(width_tiny), dtype=torch.float32),
        "dropped_invalid_width_rows": torch.tensor(dropped_invalid_width_value, dtype=torch.float32),
        "rel_denom_eps": torch.tensor(float(rel_denom_eps), dtype=torch.float32),
    }

    Xtr01 = Xtr01.to(device)
    ytr_dev = ytr.to(device)
    Xva01 = Xva01.to(device)
    yva_dev = yva.to(device)
    yva_y01_dev = yva_y01.to(device)
    Xtr_qs_dev = Xtr_raw[:, [q_idx, s_idx]].to(device)
    Xva_qs_dev = Xva_raw[:, [q_idx, s_idx]].to(device)

    model = TinyMLP_Logit(
        in_dim=Xtr01.shape[1],
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=int(y.shape[1]),
    ).to(device=device, dtype=torch.float32)

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("Enabled torch.compile(model)")
        except Exception as exc:
            print(f"Note: torch.compile unavailable/failed ({exc}); continuing without it.")

    opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
    if device.type == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
            print("Using fused AdamW")
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            print("Using AdamW (non-fused)")
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=25,
        threshold=1e-6,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
    )

    best_rel_rmse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    bad = 0
    start_epoch = 1

    if checkpoint_every and checkpoint_every > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    gen_ep = (
        torch.Generator(device="cuda").manual_seed(seed)
        if device.type == "cuda"
        else torch.Generator().manual_seed(seed)
    )
    resume_checkpoint_path = _resolve_resume_checkpoint(resume_checkpoint)
    if resume_checkpoint_path is not None:
        restored = _load_training_checkpoint(
            resume_checkpoint_path,
            model,
            opt,
            scheduler,
            strict=resume_strict,
        )
        start_epoch = restored["last_epoch"] + 1
        best_epoch = restored["best_epoch"]
        best_rel_rmse = restored["best_rel_rmse"]
        if isinstance(restored["best_state"], dict):
            best_state = restored["best_state"]
        rng = restored.get("rng")
        if isinstance(rng, dict):
            try:
                if "torch_rng_state" in rng:
                    torch.set_rng_state(rng["torch_rng_state"])
                if device.type == "cuda" and "cuda_rng_state_all" in rng:
                    torch.cuda.set_rng_state_all(rng["cuda_rng_state_all"])
                if "python_random_state" in rng:
                    random.setstate(rng["python_random_state"])
            except Exception as exc:
                print(f"Note: failed to fully restore RNG state from checkpoint: {exc}")
        gen_ep_state = restored.get("gen_ep_state")
        if gen_ep_state is not None:
            try:
                gen_ep.set_state(gen_ep_state)
            except Exception as exc:
                print(f"Note: failed to restore per-epoch generator state: {exc}")
        print(
            f"Resumed training from {resume_checkpoint_path!r} at epoch {start_epoch} "
            f"(best epoch={best_epoch}, best rel RMSE={best_rel_rmse:.6e})"
        )

    clamp_min = float(y_eps)
    clamp_max = 1.0 - clamp_min
    Ntr = int(Xtr01.shape[0])
    bs = min(max(1, int(batch_size)), Ntr)
    fmt = lambda v: f"{float(v):.6e}"

    log_exists = os.path.exists(log_path)
    log_mode = "a" if resume_checkpoint_path is not None else "w"
    with open(log_path, log_mode, encoding="utf-8") as flog:
        if (log_mode == "w") or (not log_exists):
            flog.write("# run metadata columns\n")
            flog.write("# 1: seed\n# 2: perm_sha1_16\n# 3: q_idx\n# 4: s_idx\n")
            flog.write("# 5: target_mode\n# 6: y_eps\n# 7: width_tiny\n# 8: tr_clip_frac\n# 9: va_clip_frac\n")
            flog.write("# 10: rel_denom_eps\n# 11: rel_loss_mode\n")
            flog.write(
                f"# {seed} {perm_hash} {q_idx} {s_idx} {target_mode} {y_eps:.9g} {width_tiny:.9g} "
                f"{tr_clip_frac:.6e} {va_clip_frac:.6e} {rel_denom_eps:.9g} {rel_loss_mode}\n"
            )
            if target_mode == "rho_T_W":
                flog.write("# epoch data columns\n")
                flog.write("# 1: epoch\n# 2: val_rel_mse_combined\n# 3: val_rel_MAE_rho\n# 4: val_rel_RMSE_rho\n")
                flog.write("# 5: best_rel_mse_combined\n# 6: best_epoch\n# 7: train_rel_loss\n")
                flog.write("# 8: val_mse_y01\n# 9: val_clip_frac\n# 10: lr\n")
                flog.write("# 11: val_rel_MAE_T\n# 12: val_rel_RMSE_T\n# 13: val_rel_MAE_W\n")
                flog.write("# 14: val_rel_RMSE_W\n")
            else:
                flog.write("# epoch data columns\n")
                flog.write("# 1: epoch\n# 2: val_rel_mse_x\n# 3: val_rel_MAE_x\n# 4: val_rel_RMSE_x\n")
                flog.write("# 5: best_rel_mse_x\n# 6: best_epoch\n# 7: train_rel_loss\n")
                flog.write("# 8: val_mse_y01\n# 9: val_clip_frac\n# 10: lr\n")
        if target_mode == "rho_T_W":
            print(
                "#  val_rel_mse_combined val_rel_MAE_rho val_rel_RMSE_rho best_rel_mse_combined best_epoch "
                "train_rel_loss val_mse_y01 val_clip_frac lr val_rel_MAE_T val_rel_RMSE_T "
                "val_rel_MAE_W val_rel_RMSE_W"
            )
        else:
            print(
                "#  val_rel_mse_x val_rel_MAE_x val_rel_RMSE_x best_rel_mse_x best_epoch "
                "train_rel_loss val_mse_y01 val_clip_frac lr"
            )

        for ep in range(start_epoch, epochs + 1):
            model.train()
            perm_tr = torch.randperm(Ntr, generator=gen_ep, device=Xtr01.device)
            loss_sum = torch.zeros((), device=device)
            n_batches = 0

            for start in range(0, Ntr, bs):
                idx = perm_tr[start : start + bs]
                xb01 = Xtr01[idx]
                yb = ytr_dev[idx]

                opt.zero_grad(set_to_none=True)
                y01_pred = torch.sigmoid(model(xb01)).to(torch.float32)
                if target_mode == "rho_T_W":
                    y_log_pred = y01_to_linear(y01_pred, y_out_stats, y_eps=y_eps)
                    y_pred = inverse_transform_positive_target(y_log_pred)
                    loss_rho = relative_error_loss_x(
                        y_pred[:, 0:1], yb[:, 0:1], denom_eps=rel_denom_eps, mode=rel_loss_mode
                    )
                    loss_T = relative_error_loss_x(
                        y_pred[:, 1:2], yb[:, 1:2], denom_eps=rel_denom_eps, mode=rel_loss_mode
                    )
                    loss_W = relative_error_loss_x(
                        y_pred[:, 2:3], yb[:, 2:3], denom_eps=rel_denom_eps, mode=rel_loss_mode
                    )
                    loss = (loss_rho + loss_T + loss_W) / 3.0
                else:
                    x_pred = y01_to_x_differentiable_from_qs(
                        Xtr_qs_dev[idx], y01_pred, y_eps=y_eps
                    )
                    loss = relative_error_loss_x(
                        x_pred, yb, denom_eps=rel_denom_eps, mode=rel_loss_mode
                    )
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(grad_clip_norm)
                    )
                opt.step()
                loss_sum += loss.detach()
                n_batches += 1

            train_rel_loss = (loss_sum / max(1, n_batches)).item()
            model.eval()
            with torch.no_grad():
                y01_va = torch.sigmoid(model(Xva01)).to(torch.float32)
                y01_va_pred = torch.clamp(y01_va, min=clamp_min, max=clamp_max)
                val_mse_y01 = float(((y01_va_pred - yva_y01_dev) ** 2).mean().item())
                if target_mode == "rho_T_W":
                    y_va = inverse_transform_positive_target(
                        y01_to_linear(y01_va, y_out_stats, y_eps=y_eps)
                    )
                    rho_rel = x_relative_metrics(
                        y_va[:, 0:1], yva_dev[:, 0:1], denom_eps=rel_denom_eps
                    )
                    T_rel = x_relative_metrics(
                        y_va[:, 1:2], yva_dev[:, 1:2], denom_eps=rel_denom_eps
                    )
                    W_rel = x_relative_metrics(
                        y_va[:, 2:3], yva_dev[:, 2:3], denom_eps=rel_denom_eps
                    )
                    val_rel_mse_combined = (rho_rel[0] + T_rel[0] + W_rel[0]) / 3.0
                else:
                    x_va = y01_to_x_differentiable_from_qs(
                        Xva_qs_dev, y01_va, y_eps=y_eps
                    )
                    x_rel = x_relative_metrics(
                        x_va, yva_dev, denom_eps=rel_denom_eps
                    )
                    val_rel_mse_combined = x_rel[0]

            scheduler.step(val_rel_mse_combined)
            improved = (best_rel_rmse - val_rel_mse_combined) > float(min_delta)
            if improved:
                best_rel_rmse = float(val_rel_mse_combined)
                best_epoch = ep
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1

            ep_s = f"{ep:05d}"
            be_s = f"{best_epoch:05d}"
            lr_now = float(opt.param_groups[0].get("lr", lr))
            if target_mode == "rho_T_W":
                flog.write(
                    f"{ep_s} {fmt(val_rel_mse_combined)} {fmt(rho_rel[1])} {fmt(rho_rel[2])} "
                    f"{fmt(best_rel_rmse)} {be_s} {fmt(train_rel_loss)} {fmt(val_mse_y01)} "
                    f"{fmt(va_clip_frac)} {fmt(lr_now)} {fmt(T_rel[1])} {fmt(T_rel[2])} "
                    f"{fmt(W_rel[1])} {fmt(W_rel[2])}\n"
                )
            else:
                flog.write(
                    f"{ep_s} {fmt(val_rel_mse_combined)} {fmt(x_rel[1])} {fmt(x_rel[2])} "
                    f"{fmt(best_rel_rmse)} {be_s} {fmt(train_rel_loss)} {fmt(val_mse_y01)} "
                    f"{fmt(va_clip_frac)} {fmt(lr_now)}\n"
                )
            if ep == 1 or (ep % 10 == 0):
                flog.flush()
            if target_mode == "rho_T_W":
                print(
                    "{ep} {vrm} {vra} {vrr} {br} {be} "
                    "{trl} {vmy} {cf} {lr} {vtma} {vtmr} {vwma} {vwmr}".format(
                        ep=ep_s,
                        vrm=fmt(val_rel_mse_combined),
                        vra=fmt(rho_rel[1]),
                        vrr=fmt(rho_rel[2]),
                        br=fmt(best_rel_rmse),
                        be=be_s,
                        trl=fmt(train_rel_loss),
                        vmy=fmt(val_mse_y01),
                        cf=fmt(va_clip_frac),
                        lr=fmt(lr_now),
                        vtma=fmt(T_rel[1]),
                        vtmr=fmt(T_rel[2]),
                        vwma=fmt(W_rel[1]),
                        vwmr=fmt(W_rel[2]),
                    )
                )
            else:
                print(
                    "{ep} {vrm} {vra} {vrr} {br} {be} {trl} {vmy} {cf} {lr}".format(
                        ep=ep_s,
                        vrm=fmt(val_rel_mse_combined),
                        vra=fmt(x_rel[1]),
                        vrr=fmt(x_rel[2]),
                        br=fmt(best_rel_rmse),
                        be=be_s,
                        trl=fmt(train_rel_loss),
                        vmy=fmt(val_mse_y01),
                        cf=fmt(va_clip_frac),
                        lr=fmt(lr_now),
                    )
                )

            if checkpoint_every and checkpoint_every > 0 and (ep % checkpoint_every == 0):
                ckpt_path = os.path.join(
                    checkpoint_dir, f"{checkpoint_prefix}_ep{ep:05d}.pt"
                )
                torch.save(
                    {
                        "epoch": ep,
                        "best_epoch": best_epoch,
                        "best_rel_rmse_combined": best_rel_rmse,
                        "perm_sha1_16": perm_hash,
                        "model_state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "best_model_state_dict": best_state,
                        "opt_state_dict": _optimizer_state_to_cpu(opt.state_dict()),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "lr": lr_now,
                        "scaler_state_dict": None,
                        "rng": _get_rng_snapshot(device, gen),
                        "gen_ep_state": gen_ep.get_state(),
                        "backend_flags": _get_backend_flags_snapshot(),
                        "ft_stats": {
                            "kind": ft_stats.kind.detach().cpu().to(torch.int32),
                            "eps": float(ft_stats.eps),
                            "q_idx": int(ft_stats.q_idx),
                            "s_idx": int(ft_stats.s_idx),
                        },
                        "x_stats": {k: v.detach().cpu().to(torch.float32) for k, v in x_stats.items()},
                        "y_stats": {k: v.detach().cpu().to(torch.float32) for k, v in y_stats.items()},
                        "train_rel_loss_mode": rel_loss_mode,
                        "rel_denom_eps": float(rel_denom_eps),
                    },
                    ckpt_path,
                )
                if checkpoint_export_header:
                    hdr_path = os.path.join(
                        checkpoint_dir, f"{checkpoint_prefix}_ep{ep:05d}.h"
                    )
                    export_to_c_header(model, ft_stats, x_stats, y_stats, hdr_path)
                print(f"[checkpoint] wrote {ckpt_path}")

            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ft_stats = FeatureTransformStats(
        kind=ft_stats.kind.detach().cpu().to(torch.int32),
        eps=float(ft_stats.eps),
        q_idx=int(q_idx),
        s_idx=int(s_idx),
    )
    x_stats = {k: v.detach().cpu().to(torch.float32) for k, v in x_stats.items()}
    y_stats = {k: v.detach().cpu().to(torch.float32) for k, v in y_stats.items()}
    if return_model_on_cpu:
        model = model.to(device="cpu", dtype=torch.float32)
    return model, ft_stats, x_stats, y_stats, device


def _prepare_solver_features(
    X_raw: torch.Tensor,
    *,
    q_idx: int,
    s_idx: int,
    pos_log10_ratio_thresh: float,
    neg_frac_tol: float,
    min_pos_count: int,
    force_kind: Optional[Dict[int, int]],
    verbose_transforms: bool,
    robust_mm_qlo: float,
    robust_mm_qhi: float,
) -> tuple[
    FeatureTransformStats,
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    float,
    float,
]:
    ft_stats = fit_feature_transform(
        X_raw,
        pos_log10_ratio_thresh=pos_log10_ratio_thresh,
        eps=1e-30,
        neg_frac_tol=neg_frac_tol,
        min_pos_count=min_pos_count,
        force_kind=force_kind,
        verbose=verbose_transforms,
    )
    ft_stats.q_idx = int(q_idx)
    ft_stats.s_idx = int(s_idx)
    Xt = apply_feature_transform(X_raw, ft_stats)
    mm_stats = fit_robust_minmax(Xt, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi)
    X01, clip_frac = apply_robust_minmax(Xt, mm_stats)
    x_std_stats = fit_standardizer(Xt)
    x_stats: Dict[str, torch.Tensor] = {
        "lo": mm_stats["lo"].detach().cpu().to(torch.float32),
        "hi": mm_stats["hi"].detach().cpu().to(torch.float32),
        "invrng": mm_stats["invrng"].detach().cpu().to(torch.float32),
        "mean": x_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": x_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": x_std_stats["invstd"].detach().cpu().to(torch.float32),
    }
    return ft_stats, x_stats, Xt, X01, float(clip_frac), float(clip_frac)


def train_solver_region(
    dataset: Dict[str, Any],
    *,
    hidden_dim: int = 3,
    n_hidden: int = 3,
    lr: float = 5e-4,
    epochs: int = 2000,
    batch_size: int = 4096,
    val_frac: float = 0.2,
    seed: int = 0,
    patience: int = 250,
    min_delta: float = 0.0,
    weight_decay: float = 1e-6,
    grad_clip_norm: Optional[float] = 1.0,
    return_model_on_cpu: bool = True,
    log_path: str = "training_log.txt",
    pos_log10_ratio_thresh: float = 1e3,
    robust_mm_qlo: float = 0.001,
    robust_mm_qhi: float = 0.999,
    neg_frac_tol: float = 1e-4,
    min_pos_count: int = 256,
    force_kind: Optional[Dict[int, int]] = None,
    y_eps: float = 1e-7,
    q_idx: int = 0,
    s_idx: int = 2,
    checkpoint_every: int = 200,
    checkpoint_dir: str = "checkpoints",
    checkpoint_prefix: str = "solver_region",
    verbose_transforms: bool = True,
    **_ignored,
):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    random.seed(seed)
    device = pick_device()

    X_raw = torch.tensor(dataset["X"], dtype=torch.float32)
    y_grid = torch.tensor(dataset["y_grid"], dtype=torch.float32)
    robust_mask = torch.tensor(dataset["y_robust_mask"], dtype=torch.bool)
    fast_mask = torch.tensor(dataset["y_fast_mask"], dtype=torch.bool)
    sample_weight = torch.tensor(dataset["sample_weight"], dtype=torch.float32)
    y_exact = torch.tensor(dataset["y_exact"], dtype=torch.float32).reshape(-1, 1)

    N = int(X_raw.shape[0])
    n_val = max(1, min(int(val_frac * N), N - 1))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    perm_hash = hashlib.sha1(perm.cpu().numpy().tobytes()).hexdigest()[:16]
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr_raw = X_raw[tr_idx]
    Xva_raw = X_raw[val_idx]
    ytr_exact = y_exact[tr_idx]
    yva_exact = y_exact[val_idx]
    robust_tr = robust_mask[tr_idx]
    robust_va = robust_mask[val_idx]
    fast_tr = fast_mask[tr_idx]
    fast_va = fast_mask[val_idx]
    w_tr = sample_weight[tr_idx]
    w_va = sample_weight[val_idx]

    ft_stats = fit_feature_transform(
        Xtr_raw,
        pos_log10_ratio_thresh=pos_log10_ratio_thresh,
        eps=1e-30,
        neg_frac_tol=neg_frac_tol,
        min_pos_count=min_pos_count,
        force_kind=force_kind,
        verbose=verbose_transforms,
    )
    ft_stats.q_idx = int(q_idx)
    ft_stats.s_idx = int(s_idx)
    Xtr_t = apply_feature_transform(Xtr_raw, ft_stats)
    Xva_t = apply_feature_transform(Xva_raw, ft_stats)
    mm_stats = fit_robust_minmax(Xtr_t, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi)
    Xtr01, tr_clip_frac = apply_robust_minmax(Xtr_t, mm_stats)
    Xva01, va_clip_frac = apply_robust_minmax(Xva_t, mm_stats)
    x_std_stats = fit_standardizer(Xtr_t)
    x_stats: Dict[str, torch.Tensor] = {
        "lo": mm_stats["lo"].detach().cpu().to(torch.float32),
        "hi": mm_stats["hi"].detach().cpu().to(torch.float32),
        "invrng": mm_stats["invrng"].detach().cpu().to(torch.float32),
        "mean": x_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": x_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": x_std_stats["invstd"].detach().cpu().to(torch.float32),
    }
    y_stats: Dict[str, torch.Tensor] = {
        "mean": ytr_exact.mean(dim=0).detach().cpu().to(torch.float32),
        "std": ytr_exact.std(dim=0, unbiased=False).detach().cpu().to(torch.float32),
        "invstd": torch.where(
            ytr_exact.std(dim=0, unbiased=False) < 1e-8,
            torch.ones_like(ytr_exact.std(dim=0, unbiased=False)),
            1.0 / ytr_exact.std(dim=0, unbiased=False),
        ).detach().cpu().to(torch.float32),
        "out_kind": torch.tensor([OUT_KIND_X_BOUNDED], dtype=torch.int32),
        "out_lo": torch.zeros(1, dtype=torch.float32),
        "out_hi": torch.ones(1, dtype=torch.float32),
        "out_invrng": torch.ones(1, dtype=torch.float32),
        "y_eps": torch.tensor(float(y_eps), dtype=torch.float32),
        "width_tiny": torch.tensor(1.0e-12, dtype=torch.float32),
        "robust_mm_qlo": torch.tensor(float(robust_mm_qlo), dtype=torch.float32),
        "robust_mm_qhi": torch.tensor(float(robust_mm_qhi), dtype=torch.float32),
        "pos_log10_ratio_thresh": torch.tensor(float(pos_log10_ratio_thresh), dtype=torch.float32),
        "neg_frac_tol": torch.tensor(float(neg_frac_tol), dtype=torch.float32),
        "min_pos_count": torch.tensor(float(min_pos_count), dtype=torch.float32),
        "dropped_invalid_width_rows": torch.tensor(0.0, dtype=torch.float32),
        "rel_denom_eps": torch.tensor(1.0e-12, dtype=torch.float32),
        "solver_target_mode": torch.tensor(1.0, dtype=torch.float32),
    }

    Xtr01 = Xtr01.to(device)
    Xva01 = Xva01.to(device)
    robust_tr = robust_tr.to(device)
    robust_va = robust_va.to(device)
    fast_tr = fast_tr.to(device)
    fast_va = fast_va.to(device)
    w_tr = w_tr.to(device)
    w_va = w_va.to(device)
    y_grid_dev = y_grid.to(device)
    ytr_exact = ytr_exact.to(device)
    yva_exact = yva_exact.to(device)

    model = TinyMLP_Logit(
        in_dim=Xtr01.shape[1],
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=1,
    ).to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("Enabled torch.compile(model)")
        except Exception as exc:
            print(f"Note: torch.compile unavailable/failed ({exc}); continuing without it.")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=25, threshold=1e-6, threshold_mode="rel", min_lr=1e-6
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    Ntr = int(Xtr01.shape[0])
    bs = min(max(1, int(batch_size)), Ntr)

    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("# epoch val_region_loss best_val best_epoch train_region_loss val_hit_robust val_hit_fast lr\n")
        print("# epoch val_region_loss best_val best_epoch train_region_loss val_hit_robust val_hit_fast lr")
        for ep in range(1, epochs + 1):
            model.train()
            perm_tr = torch.randperm(Ntr, device=Xtr01.device)
            loss_sum = torch.zeros((), device=device)
            n_batches = 0
            for start in range(0, Ntr, bs):
                idx = perm_tr[start : start + bs]
                opt.zero_grad(set_to_none=True)
                y_pred = torch.sigmoid(model(Xtr01[idx])).to(torch.float32)
                loss = region_distance_loss(
                    y_pred,
                    y_grid_dev,
                    robust_tr[idx],
                    fast_tr[idx],
                    w_tr[idx],
                )
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()
                loss_sum += loss.detach()
                n_batches += 1

            train_loss = float((loss_sum / max(1, n_batches)).item())
            model.eval()
            with torch.no_grad():
                yva_pred = torch.sigmoid(model(Xva01)).to(torch.float32)
                val_loss = float(
                    region_distance_loss(
                        yva_pred, y_grid_dev, robust_va, fast_va, w_va
                    ).item()
                )
                val_hit_robust = y01_to_region_hit_rate(yva_pred, y_grid_dev, robust_va)
                val_hit_fast = y01_to_region_hit_rate(yva_pred, y_grid_dev, fast_va)
                val_y_mae = float((yva_pred - yva_exact).abs().mean().item())

            scheduler.step(val_loss)
            improved = (best_val - val_loss) > float(min_delta)
            if improved:
                best_val = val_loss
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            lr_now = float(opt.param_groups[0].get("lr", lr))
            ep_s = f"{ep:05d}"
            line = (
                f"{ep_s} {val_loss:.6e} {best_val:.6e} {best_epoch:05d} "
                f"{train_loss:.6e} {val_hit_robust:.6e} {val_hit_fast:.6e} {lr_now:.6e}\n"
            )
            flog.write(line)
            if ep == 1 or (ep % 10 == 0):
                flog.flush()
            print(line.strip())

            if checkpoint_every and checkpoint_every > 0 and (ep % checkpoint_every == 0):
                ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_ep{ep:05d}.pt")
                torch.save(
                    {
                        "epoch": ep,
                        "best_epoch": best_epoch,
                        "best_region_loss": best_val,
                        "perm_sha1_16": perm_hash,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "best_model_state_dict": best_state,
                        "x_stats": {k: v.detach().cpu().to(torch.float32) for k, v in x_stats.items()},
                        "y_stats": {k: v.detach().cpu().to(torch.float32) for k, v in y_stats.items()},
                        "ft_stats": {
                            "kind": ft_stats.kind.detach().cpu().to(torch.int32),
                            "eps": float(ft_stats.eps),
                            "q_idx": int(ft_stats.q_idx),
                            "s_idx": int(ft_stats.s_idx),
                        },
                        "val_y_mae": val_y_mae,
                    },
                    ckpt_path,
                )
                print(f"[checkpoint] wrote {ckpt_path}")

            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if return_model_on_cpu:
        model = model.to(device="cpu", dtype=torch.float32)
    return model, ft_stats, x_stats, y_stats, device


def train_solver_soft_bins(
    dataset: Dict[str, Any],
    *,
    hidden_dim: int = 3,
    n_hidden: int = 3,
    lr: float = 5e-4,
    epochs: int = 2000,
    batch_size: int = 4096,
    val_frac: float = 0.2,
    seed: int = 0,
    patience: int = 250,
    min_delta: float = 0.0,
    weight_decay: float = 1e-6,
    grad_clip_norm: Optional[float] = 1.0,
    return_model_on_cpu: bool = True,
    log_path: str = "training_log.txt",
    pos_log10_ratio_thresh: float = 1e3,
    robust_mm_qlo: float = 0.001,
    robust_mm_qhi: float = 0.999,
    neg_frac_tol: float = 1e-4,
    min_pos_count: int = 256,
    force_kind: Optional[Dict[int, int]] = None,
    q_idx: int = 0,
    s_idx: int = 2,
    verbose_transforms: bool = True,
    **_ignored,
):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    random.seed(seed)
    device = pick_device()

    X_raw = torch.tensor(dataset["X"], dtype=torch.float32)
    y_grid = torch.tensor(dataset["y_grid"], dtype=torch.float32)
    soft_prob = torch.tensor(dataset["soft_prob"], dtype=torch.float32)
    sample_weight = torch.tensor(dataset["sample_weight"], dtype=torch.float32)

    N = int(X_raw.shape[0])
    n_val = max(1, min(int(val_frac * N), N - 1))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr_raw = X_raw[tr_idx]
    Xva_raw = X_raw[val_idx]
    ptr = soft_prob[tr_idx]
    pva = soft_prob[val_idx]
    wtr = sample_weight[tr_idx]
    wva = sample_weight[val_idx]

    ft_stats = fit_feature_transform(
        Xtr_raw,
        pos_log10_ratio_thresh=pos_log10_ratio_thresh,
        eps=1e-30,
        neg_frac_tol=neg_frac_tol,
        min_pos_count=min_pos_count,
        force_kind=force_kind,
        verbose=verbose_transforms,
    )
    ft_stats.q_idx = int(q_idx)
    ft_stats.s_idx = int(s_idx)
    Xtr_t = apply_feature_transform(Xtr_raw, ft_stats)
    Xva_t = apply_feature_transform(Xva_raw, ft_stats)
    mm_stats = fit_robust_minmax(Xtr_t, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi)
    Xtr01, _ = apply_robust_minmax(Xtr_t, mm_stats)
    Xva01, _ = apply_robust_minmax(Xva_t, mm_stats)
    x_std_stats = fit_standardizer(Xtr_t)
    x_stats: Dict[str, torch.Tensor] = {
        "lo": mm_stats["lo"].detach().cpu().to(torch.float32),
        "hi": mm_stats["hi"].detach().cpu().to(torch.float32),
        "invrng": mm_stats["invrng"].detach().cpu().to(torch.float32),
        "mean": x_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": x_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": x_std_stats["invstd"].detach().cpu().to(torch.float32),
    }
    y_stats: Dict[str, torch.Tensor] = {
        "mean": ptr.mean(dim=0).detach().cpu().to(torch.float32),
        "std": ptr.std(dim=0, unbiased=False).detach().cpu().to(torch.float32),
        "invstd": torch.ones(ptr.shape[1], dtype=torch.float32),
        "out_kind": torch.full((ptr.shape[1],), OUT_KIND_LINEAR, dtype=torch.int32),
        "out_lo": torch.zeros(ptr.shape[1], dtype=torch.float32),
        "out_hi": torch.ones(ptr.shape[1], dtype=torch.float32),
        "out_invrng": torch.ones(ptr.shape[1], dtype=torch.float32),
        "solver_soft_bins": torch.tensor(1.0, dtype=torch.float32),
        "solver_y_grid": y_grid.detach().cpu().to(torch.float32),
    }

    Xtr01 = Xtr01.to(device)
    Xva01 = Xva01.to(device)
    ptr = ptr.to(device)
    pva = pva.to(device)
    wtr = wtr.to(device)
    wva = wva.to(device)

    model = TinyMLP_Logit(
        in_dim=Xtr01.shape[1],
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=int(ptr.shape[1]),
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=25, threshold=1e-6, threshold_mode="rel", min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    Ntr = int(Xtr01.shape[0])
    bs = min(max(1, int(batch_size)), Ntr)
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("# epoch val_soft_ce best_val best_epoch train_soft_ce val_top1_acc lr\n")
        print("# epoch val_soft_ce best_val best_epoch train_soft_ce val_top1_acc lr")
        for ep in range(1, epochs + 1):
            model.train()
            perm_tr = torch.randperm(Ntr, device=Xtr01.device)
            loss_sum = torch.zeros((), device=device)
            n_batches = 0
            for start in range(0, Ntr, bs):
                idx = perm_tr[start : start + bs]
                opt.zero_grad(set_to_none=True)
                logits = model(Xtr01[idx]).to(torch.float32)
                loss = soft_target_cross_entropy(logits, ptr[idx], wtr[idx])
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()
                loss_sum += loss.detach()
                n_batches += 1

            train_loss = float((loss_sum / max(1, n_batches)).item())
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva01).to(torch.float32)
                val_loss = float(soft_target_cross_entropy(logits_va, pva, wva).item())
                pred_idx = torch.argmax(logits_va, dim=1)
                true_idx = torch.argmax(pva, dim=1)
                val_top1_acc = float((pred_idx == true_idx).float().mean().item())
            scheduler.step(val_loss)
            improved = (best_val - val_loss) > float(min_delta)
            if improved:
                best_val = val_loss
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            lr_now = float(opt.param_groups[0].get("lr", lr))
            line = (
                f"{ep:05d} {val_loss:.6e} {best_val:.6e} {best_epoch:05d} "
                f"{train_loss:.6e} {val_top1_acc:.6e} {lr_now:.6e}\n"
            )
            flog.write(line)
            if ep == 1 or (ep % 10 == 0):
                flog.flush()
            print(line.strip())
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if return_model_on_cpu:
        model = model.to(device="cpu", dtype=torch.float32)
    return model, ft_stats, x_stats, y_stats, device


def train_solver_gate(
    dataset: Dict[str, Any],
    *,
    hidden_dim: int = 3,
    n_hidden: int = 3,
    lr: float = 5e-4,
    epochs: int = 2000,
    batch_size: int = 4096,
    val_frac: float = 0.2,
    seed: int = 0,
    patience: int = 250,
    min_delta: float = 0.0,
    weight_decay: float = 1e-6,
    grad_clip_norm: Optional[float] = 1.0,
    return_model_on_cpu: bool = True,
    log_path: str = "training_log.txt",
    pos_log10_ratio_thresh: float = 1e3,
    robust_mm_qlo: float = 0.001,
    robust_mm_qhi: float = 0.999,
    neg_frac_tol: float = 1e-4,
    min_pos_count: int = 256,
    force_kind: Optional[Dict[int, int]] = None,
    q_idx: int = 0,
    s_idx: int = 2,
    verbose_transforms: bool = True,
    **_ignored,
):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    random.seed(seed)
    device = pick_device()

    X_raw = torch.tensor(dataset["X"], dtype=torch.float32)
    gate_target = torch.tensor(dataset["gate_target"], dtype=torch.float32).reshape(-1, 1)
    sample_weight = torch.tensor(dataset["sample_weight"], dtype=torch.float32).reshape(-1, 1)

    N = int(X_raw.shape[0])
    n_val = max(1, min(int(val_frac * N), N - 1))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr_raw = X_raw[tr_idx]
    Xva_raw = X_raw[val_idx]
    ytr = gate_target[tr_idx]
    yva = gate_target[val_idx]
    wtr = sample_weight[tr_idx]
    wva = sample_weight[val_idx]

    ft_stats = fit_feature_transform(
        Xtr_raw,
        pos_log10_ratio_thresh=pos_log10_ratio_thresh,
        eps=1e-30,
        neg_frac_tol=neg_frac_tol,
        min_pos_count=min_pos_count,
        force_kind=force_kind,
        verbose=verbose_transforms,
    )
    ft_stats.q_idx = int(q_idx)
    ft_stats.s_idx = int(s_idx)
    Xtr_t = apply_feature_transform(Xtr_raw, ft_stats)
    Xva_t = apply_feature_transform(Xva_raw, ft_stats)
    mm_stats = fit_robust_minmax(Xtr_t, q_lo=robust_mm_qlo, q_hi=robust_mm_qhi)
    Xtr01, _ = apply_robust_minmax(Xtr_t, mm_stats)
    Xva01, _ = apply_robust_minmax(Xva_t, mm_stats)
    x_std_stats = fit_standardizer(Xtr_t)
    x_stats: Dict[str, torch.Tensor] = {
        "lo": mm_stats["lo"].detach().cpu().to(torch.float32),
        "hi": mm_stats["hi"].detach().cpu().to(torch.float32),
        "invrng": mm_stats["invrng"].detach().cpu().to(torch.float32),
        "mean": x_std_stats["mean"].detach().cpu().to(torch.float32),
        "std": x_std_stats["std"].detach().cpu().to(torch.float32),
        "invstd": x_std_stats["invstd"].detach().cpu().to(torch.float32),
    }
    y_stats: Dict[str, torch.Tensor] = {
        "mean": ytr.mean(dim=0).detach().cpu().to(torch.float32),
        "std": ytr.std(dim=0, unbiased=False).detach().cpu().to(torch.float32),
        "invstd": torch.ones(1, dtype=torch.float32),
        "out_kind": torch.tensor([OUT_KIND_LINEAR], dtype=torch.int32),
        "out_lo": torch.zeros(1, dtype=torch.float32),
        "out_hi": torch.ones(1, dtype=torch.float32),
        "out_invrng": torch.ones(1, dtype=torch.float32),
        "solver_gate": torch.tensor(1.0, dtype=torch.float32),
    }

    Xtr01 = Xtr01.to(device)
    Xva01 = Xva01.to(device)
    ytr = ytr.to(device)
    yva = yva.to(device)
    wtr = wtr.to(device)
    wva = wva.to(device)

    model = TinyMLP_Logit(
        in_dim=Xtr01.shape[1],
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=1,
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=25, threshold=1e-6, threshold_mode="rel", min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    Ntr = int(Xtr01.shape[0])
    bs = min(max(1, int(batch_size)), Ntr)
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("# epoch val_bce best_val best_epoch train_bce val_acc lr\n")
        print("# epoch val_bce best_val best_epoch train_bce val_acc lr")
        for ep in range(1, epochs + 1):
            model.train()
            perm_tr = torch.randperm(Ntr, device=Xtr01.device)
            loss_sum = torch.zeros((), device=device)
            n_batches = 0
            for start in range(0, Ntr, bs):
                idx = perm_tr[start : start + bs]
                opt.zero_grad(set_to_none=True)
                logits = model(Xtr01[idx]).to(torch.float32)
                loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, ytr[idx], reduction="none"
                )
                loss = (loss_raw * wtr[idx]).sum() / torch.clamp(wtr[idx].sum(), min=1.0)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                opt.step()
                loss_sum += loss.detach()
                n_batches += 1

            train_loss = float((loss_sum / max(1, n_batches)).item())
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva01).to(torch.float32)
                loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_va, yva, reduction="none"
                )
                val_loss = float((loss_raw * wva).sum().item() / max(1.0, float(wva.sum().item())))
                pred = (torch.sigmoid(logits_va) >= 0.5).to(torch.float32)
                val_acc = float((pred == yva).float().mean().item())
            scheduler.step(val_loss)
            improved = (best_val - val_loss) > float(min_delta)
            if improved:
                best_val = val_loss
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            lr_now = float(opt.param_groups[0].get("lr", lr))
            line = (
                f"{ep:05d} {val_loss:.6e} {best_val:.6e} {best_epoch:05d} "
                f"{train_loss:.6e} {val_acc:.6e} {lr_now:.6e}\n"
            )
            flog.write(line)
            if ep == 1 or (ep % 10 == 0):
                flog.flush()
            print(line.strip())
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if return_model_on_cpu:
        model = model.to(device="cpu", dtype=torch.float32)
    return model, ft_stats, x_stats, y_stats, device


@torch.no_grad()
def export_to_c_header(
    model: nn.Module,
    ft_stats: FeatureTransformStats,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
    path: str,
):
    if not (hasattr(model, "fcs") and hasattr(model, "out") and hasattr(model, "n_hidden")):
        raise TypeError(
            "export_to_c_header expects a TinyMLP_Logit-like model with .fcs and .out."
        )

    sd = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    if "fcs.0.weight" not in sd:
        for prefix in ("_orig_mod.", "module."):
            if (prefix + "fcs.0.weight") in sd:
                sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
                break

    W_in = sd["fcs.0.weight"]
    b_in = sd["fcs.0.bias"]
    H, Din = W_in.shape
    n_hidden = int(model.n_hidden)
    W_hid = [sd[f"fcs.{li}.weight"] for li in range(1, n_hidden)]
    b_hid = [sd[f"fcs.{li}.bias"] for li in range(1, n_hidden)]
    W_out = sd["out.weight"]
    b_out = sd["out.bias"]

    x_kind = ft_stats.kind.detach().cpu().to(torch.int32)
    x_lo = x_stats["lo"].detach().cpu().to(torch.float32)
    x_hi = x_stats["hi"].detach().cpu().to(torch.float32)
    x_invrng = x_stats["invrng"].detach().cpu().to(torch.float32)
    out_kind = y_stats["out_kind"].detach().cpu().to(torch.int32)
    out_lo = y_stats["out_lo"].detach().cpu().to(torch.float32)
    out_hi = y_stats["out_hi"].detach().cpu().to(torch.float32)
    out_invrng = y_stats["out_invrng"].detach().cpu().to(torch.float32)
    y_eps = float(y_stats.get("y_eps", torch.tensor(1e-7)).item())
    q_idx = int(ft_stats.q_idx)
    s_idx = int(ft_stats.s_idx)

    if int(x_kind.numel()) != int(Din):
        raise ValueError(f"nn_x_kind length {x_kind.numel()} != Din {Din}")
    if any(int(t.numel()) != int(Din) for t in (x_lo, x_hi, x_invrng)):
        raise ValueError("x_stats lengths must match input dimension")
    if int(out_kind.numel()) != int(model.out_dim):
        raise ValueError(f"nn_out_kind length {out_kind.numel()} != Dout {model.out_dim}")

    def ensure_finite(name: str, t: torch.Tensor) -> None:
        if not torch.isfinite(t).all():
            raise ValueError(f"Non-finite values detected in '{name}'. Refusing to export header.")

    for name, tensor in (
        ("nn_W_in", W_in),
        ("nn_b_in", b_in),
        ("nn_W_out", W_out),
        ("nn_b_out", b_out),
        ("nn_x_lo", x_lo),
        ("nn_x_hi", x_hi),
        ("nn_x_invrng", x_invrng),
        ("nn_out_lo", out_lo),
        ("nn_out_hi", out_hi),
        ("nn_out_invrng", out_invrng),
    ):
        ensure_finite(name, tensor)
    for li, tensor in enumerate(W_hid):
        ensure_finite(f"nn_W_hid[{li}]", tensor)
    for li, tensor in enumerate(b_hid):
        ensure_finite(f"nn_b_hid[{li}]", tensor)

    def c_float32(v: float) -> str:
        if not math.isfinite(v):
            raise ValueError(f"Non-finite float encountered during export: {v}")
        if v == 0.0:
            v = 0.0
        s = f"{float(v):.9g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return s + "f"

    def as_c_array_int(name: str, t: torch.Tensor) -> str:
        vals = ", ".join(str(int(v)) for v in t.reshape(-1).tolist())
        return f"static const int {name}[{t.numel()}] = {{ {vals} }};\n"

    def as_c_array_1d(name: str, t: torch.Tensor) -> str:
        vals = ", ".join(c_float32(float(v)) for v in t.reshape(-1).tolist())
        return f"static const float {name}[{t.numel()}] = {{ {vals} }};\n"

    def as_c_array_2d(name: str, t: torch.Tensor) -> str:
        r, c = t.shape
        rows = []
        for i in range(r):
            vals = ", ".join(c_float32(float(v)) for v in t[i].tolist())
            rows.append(f"  {{ {vals} }}")
        return f"static const float {name}[{r}][{c}] = {{\n" + ",\n".join(rows) + "\n};\n"

    base = os.path.basename(path)
    guard_base = re.sub(r"[^0-9A-Za-z_]", "_", base).upper()
    guard_hash = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10].upper()
    guard = f"{guard_base}_{guard_hash}"

    header = [
        "/*============================================================================*/\n",
        "// Auto-generated tiny MLP weights/stats (float32)\n",
        "/*============================================================================*/\n",
    ]
    if int(model.out_dim) == 1:
        header.append("// Model: logit -> sigmoid -> bounded x (Palenzuela lower-bound correction)\n")
    elif int(model.out_dim) == 3:
        header.append("// Model: logit -> sigmoid -> log-scaled (rho, T, W)\n")
    else:
        header.append(f"// Model: logit -> sigmoid -> out_dim={int(model.out_dim)}\n")
    if EXPORT_USE_INCLUDE_GUARD:
        header.extend([f"#ifndef {guard}\n", f"#define {guard}\n\n"])
    else:
        header.append("\n")

    if EXPORT_ADD_AUDIT_COMMENTS:
        header.append("/* audit:\n")
        header.append(f" * q_idx={q_idx} s_idx={s_idx} y_eps={y_eps:.9g}\n")
        for key in (
            "robust_mm_qlo",
            "robust_mm_qhi",
            "pos_log10_ratio_thresh",
            "neg_frac_tol",
            "min_pos_count",
            "width_tiny",
            "dropped_invalid_width_rows",
            "rel_denom_eps",
        ):
            if key in y_stats:
                val = float(y_stats[key].reshape(-1)[0].item())
                header.append(f" * {key}={val:.9g}\n")
        header.append(" */\n\n")

    header.extend(
        [
            f"#define NN_IN_DIM {Din}\n",
            f"#define NN_HIDDEN_DIM {H}\n",
            f"#define NN_N_HIDDEN {n_hidden}\n",
            f"#define NN_OUT_DIM {model.out_dim}\n",
            f"#define NN_X_EPS {c_float32(float(ft_stats.eps))}\n",
            f"#define NN_Y_EPS {c_float32(float(y_eps))}\n",
            f"#define NN_Q_IDX {q_idx}\n",
            f"#define NN_S_IDX {s_idx}\n\n",
            as_c_array_int("nn_x_kind", x_kind),
            as_c_array_1d("nn_x_lo", x_lo),
            as_c_array_1d("nn_x_hi", x_hi),
            as_c_array_1d("nn_x_invrng", x_invrng),
            as_c_array_int("nn_out_kind", out_kind),
            as_c_array_1d("nn_out_lo", out_lo),
            as_c_array_1d("nn_out_hi", out_hi),
            as_c_array_1d("nn_out_invrng", out_invrng),
            "\n",
            as_c_array_2d("nn_W_in", W_in),
            as_c_array_1d("nn_b_in", b_in),
            "\n",
        ]
    )

    if n_hidden > 1:
        header.append(
            f"static const float nn_W_hid[{n_hidden-1}][NN_HIDDEN_DIM][NN_HIDDEN_DIM] = {{\n"
        )
        for li, W in enumerate(W_hid):
            rows = []
            for i in range(H):
                vals = ", ".join(c_float32(float(v)) for v in W[i].tolist())
                rows.append(f"    {{ {vals} }}")
            header.append(f"  {{\n" + ",\n".join(rows) + f"\n  }}{',' if li != len(W_hid)-1 else ''}\n")
        header.append("};\n\n")
        header.append(f"static const float nn_b_hid[{n_hidden-1}][NN_HIDDEN_DIM] = {{\n")
        for li, b in enumerate(b_hid):
            vals = ", ".join(c_float32(float(v)) for v in b.tolist())
            header.append(f"  {{ {vals} }}{',' if li != len(b_hid)-1 else ''}\n")
        header.append("};\n\n")

    header.extend([as_c_array_2d("nn_W_out", W_out), as_c_array_1d("nn_b_out", b_out), "\n"])
    if EXPORT_USE_INCLUDE_GUARD:
        header.append(f"#endif /* {guard} */\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(header))
    print(f"Wrote C header: {path}")


@torch.no_grad()
def export_to_hdf5(
    model: nn.Module,
    ft_stats: FeatureTransformStats,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
    path: str | Path,
    *,
    eos_path: str | Path | None = None,
):
    import numpy as np

    if not (hasattr(model, "fcs") and hasattr(model, "out") and hasattr(model, "n_hidden")):
        raise TypeError(
            "export_to_hdf5 expects a TinyMLP_Logit-like model with .fcs and .out."
        )

    sd = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    if "fcs.0.weight" not in sd:
        for prefix in ("_orig_mod.", "module."):
            if (prefix + "fcs.0.weight") in sd:
                sd = {
                    k[len(prefix) :] if k.startswith(prefix) else k: v
                    for k, v in sd.items()
                }
                break

    W_in = sd["fcs.0.weight"].numpy().astype(np.float32, copy=False)
    b_in = sd["fcs.0.bias"].numpy().astype(np.float32, copy=False)
    n_hidden = int(model.n_hidden)
    hidden_dim = int(model.hidden_dim)
    out_dim = int(model.out_dim)
    if n_hidden > 1:
        W_hid = np.stack(
            [sd[f"fcs.{li}.weight"].numpy() for li in range(1, n_hidden)],
            axis=0,
        ).astype(np.float32, copy=False)
        b_hid = np.stack(
            [sd[f"fcs.{li}.bias"].numpy() for li in range(1, n_hidden)],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        W_hid = np.zeros((0, hidden_dim, hidden_dim), dtype=np.float32)
        b_hid = np.zeros((0, hidden_dim), dtype=np.float32)
    W_out = sd["out.weight"].numpy().astype(np.float32, copy=False)
    b_out = sd["out.bias"].numpy().astype(np.float32, copy=False)

    payload = {
        "dims": {
            "in_dim": np.int32(model.in_dim),
            "hidden_dim": np.int32(hidden_dim),
            "n_hidden": np.int32(n_hidden),
            "out_dim": np.int32(out_dim),
        },
        "meta": {
            "q_idx": np.int32(ft_stats.q_idx),
            "s_idx": np.int32(ft_stats.s_idx),
            "y_eps": np.float32(float(y_stats["y_eps"].item())),
            "dx_eps": np.float32(float(y_stats["width_tiny"].item())),
        },
        "scaling": {
            "x_eps": np.float32(float(ft_stats.eps)),
            "x_kind": ft_stats.kind.detach().cpu().numpy().astype(np.int32, copy=False),
            "x_lo": x_stats["lo"].detach().cpu().numpy().astype(np.float32, copy=False),
            "x_hi": x_stats["hi"].detach().cpu().numpy().astype(np.float32, copy=False),
            "x_invrng": x_stats["invrng"].detach().cpu().numpy().astype(np.float32, copy=False),
            "out_kind": y_stats["out_kind"].detach().cpu().numpy().astype(np.int32, copy=False),
            "out_lo": y_stats["out_lo"].detach().cpu().numpy().astype(np.float32, copy=False),
            "out_hi": y_stats["out_hi"].detach().cpu().numpy().astype(np.float32, copy=False),
            "out_invrng": y_stats["out_invrng"].detach().cpu().numpy().astype(np.float32, copy=False),
        },
        "layers": {
            "W_in": W_in,
            "b_in": b_in,
            "W_hid": W_hid,
            "b_hid": b_hid,
            "W_out": W_out,
            "b_out": b_out,
        },
        "audit": {},
    }
    for key in (
        "robust_mm_qlo",
        "robust_mm_qhi",
        "pos_log10_ratio_thresh",
        "neg_frac_tol",
        "min_pos_count",
        "width_tiny",
        "dropped_invalid_width_rows",
        "rel_denom_eps",
    ):
        if key in y_stats:
            payload["audit"][key] = str(float(y_stats[key].reshape(-1)[0].item()))
    if eos_path is not None:
        payload["source_eos"] = build_eos_metadata(eos_path)
    write_nn_hdf5(payload, path)
    print(f"Wrote HDF5 model: {path}")


def train_on_dataset(
    dataset_path: str | Path,
    *,
    target_mode: str = "x_correction",
    bundle_path: str | Path | None = "tiny_mlp_inference.pt",
    hdf5_path: str | Path = "tiny_mlp_model.h5",
    header_path: str | Path | None = "tiny_mlp_weights.h",
    eos_path: str | Path | None = None,
    append_to_eos: bool = True,
    overwrite_eos_nn: bool = False,
    register_installed_model: bool = True,
    overwrite_installed_model: bool = False,
    matmul_precision: str = "high",
    **train_kwargs,
):
    if append_to_eos and eos_path is None:
        raise ValueError(
            "append_to_eos=True requires eos_path to be provided. "
            "Pass eos_path=... or set append_to_eos=False."
        )
    if eos_path is not None:
        eos_info = eos_nn_metadata(eos_path)
        if eos_info["contains_nn"]:
            print(
                f"Found existing neural-network data in {eos_path} "
                f"under group {eos_info['group_name']}."
            )
            if append_to_eos and not overwrite_eos_nn:
                raise ValueError(
                    f"{eos_path!s} already contains {eos_info['group_name']!r}. "
                    "Rerun with --overwrite_eos."
                )
        else:
            print(f"No embedded neural-network data found in {eos_path}.")
    torch.set_float32_matmul_precision(matmul_precision)

    if target_mode == "solver_region":
        dataset = read_solver_label_dataset(dataset_path, target_mode=target_mode)
        model, ft_stats, x_stats, y_stats, device = train_solver_region(
            dataset,
            **train_kwargs,
        )
    elif target_mode == "solver_soft_bins":
        if append_to_eos or register_installed_model:
            raise ValueError(
                "solver_soft_bins is a Python-only training mode for now. "
                "Use --append_eos no and --register_installed_model no."
            )
        dataset = read_solver_label_dataset(dataset_path, target_mode=target_mode)
        model, ft_stats, x_stats, y_stats, device = train_solver_soft_bins(
            dataset,
            **train_kwargs,
        )
    elif target_mode == "solver_gate":
        if append_to_eos or register_installed_model:
            raise ValueError(
                "solver_gate is a Python-only training mode for now. "
                "Use --append_eos no and --register_installed_model no."
            )
        dataset = read_solver_label_dataset(dataset_path, target_mode=target_mode)
        model, ft_stats, x_stats, y_stats, device = train_solver_gate(
            dataset,
            **train_kwargs,
        )
    else:
        data_np = read_training_dataset(dataset_path, target_mode=target_mode)
        model, ft_stats, x_stats, y_stats, device = train_regressor(
            data_np,
            target_mode=target_mode,
            **train_kwargs,
        )

    if bundle_path is not None:
        save_inference_bundle(str(bundle_path), model, ft_stats, x_stats, y_stats)

    if target_mode not in ("solver_soft_bins", "solver_gate"):
        export_to_hdf5(model, ft_stats, x_stats, y_stats, str(hdf5_path), eos_path=eos_path)
        if header_path is not None:
            export_to_c_header(model, ft_stats, x_stats, y_stats, str(header_path))
        if append_to_eos:
            summary = append_nn_to_eos_file(eos_path, hdf5_path, overwrite=overwrite_eos_nn)
            action = "Overwrote" if summary["overwrite_performed"] else "Appended"
            print(f"{action} neural-network data into EOS file: {summary['eos_filename']}")
            print(
                "Embedded network summary: "
                f"hidden_layers={summary.get('n_hidden')} "
                f"hidden_dim={summary.get('hidden_dim')} "
                f"group={summary['group_name']} "
                f"added_utc={summary.get('embedded_utc', 'unknown')}"
            )
        if register_installed_model and eos_path is not None:
            try:
                installed_path = install_nn_model(
                    hdf5_path,
                    overwrite=overwrite_installed_model,
                )
                print(f"Registered installed NN model: {installed_path}")
            except Exception as exc:
                print(f"Warning: could not register installed NN model: {exc}")
    else:
        print(f"Skipped HDF5/header/EOS export for {target_mode}; bundle export is available.")
    return model, ft_stats, x_stats, y_stats, device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("eos_file", type=Path)
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--hdf5_output", type=Path, default=Path("tiny_mlp_model.h5"))
    parser.add_argument("--bundle_output", type=Path, default=Path("tiny_mlp_inference.pt"))
    parser.add_argument("--header_output", type=Path, default=Path("tiny_mlp_weights.h"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--log_path", default="training_log.txt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--checkpoint_prefix", default="tiny_mlp")
    parser.add_argument("--hidden_dim", type=int, default=3)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append_eos", choices=("yes", "no"), default="yes")
    parser.add_argument("--overwrite_eos", action="store_true")
    parser.add_argument("--register_installed_model", choices=("yes", "no"), default="yes")
    parser.add_argument("--overwrite_installed_model", action="store_true")
    parser.add_argument("--dataset_n_pts", type=int, default=16)
    parser.add_argument(
        "--target_mode",
        choices=(
            "x_correction",
            "x_best_correction",
            "rho_T_W",
            "solver_region",
            "solver_soft_bins",
            "solver_gate",
        ),
        default="x_correction",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    append_to_eos = args.append_eos == "yes"
    register_installed_model = args.register_installed_model == "yes"
    dataset_path = args.dataset
    autogenerated_dataset: Path | None = None
    try:
        if dataset_path is None:
            if args.target_mode in ("solver_region", "solver_soft_bins", "solver_gate"):
                raise ValueError(
                    f"target_mode {args.target_mode!r} requires an explicit solver label dataset path."
                )
            fd, temp_name = tempfile.mkstemp(
                prefix=f"grhayl_nn_training_{args.eos_file.stem}_",
                suffix=".bin",
            )
            os.close(fd)
            Path(temp_name).unlink(missing_ok=True)
            autogenerated_dataset = Path(temp_name)
            print(f"Generating training dataset for {args.eos_file} ...")
            generate_dataset(
                args.eos_file,
                "train",
                n_pts=args.dataset_n_pts,
                output=autogenerated_dataset,
            )
            dataset_path = autogenerated_dataset
            print(f"Generated temporary training dataset: {dataset_path}")

        train_on_dataset(
            dataset_path,
            target_mode=args.target_mode,
            bundle_path=args.bundle_output,
            hdf5_path=args.hdf5_output,
            header_path=args.header_output,
            eos_path=args.eos_file,
            append_to_eos=append_to_eos,
            overwrite_eos_nn=args.overwrite_eos,
            register_installed_model=register_installed_model,
            overwrite_installed_model=args.overwrite_installed_model,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            lr=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            log_path=args.log_path,
            q_idx=0,
            s_idx=2,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_export_header=True,
            checkpoint_prefix=args.checkpoint_prefix,
            resume_checkpoint=args.checkpoint,
            width_tiny=1.0e-12,
            drop_invalid_width_rows=True,
            verbose_transforms=True,
            rel_denom_eps=1.0e-12,
            rel_loss_mode="mse",
        )
    finally:
        if autogenerated_dataset is not None and autogenerated_dataset.exists():
            autogenerated_dataset.unlink()
            print(f"Removed temporary training dataset: {autogenerated_dataset}")
    print("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
