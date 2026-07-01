from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch import nn

from ._nn_common import (
    FeatureTransformStats,
    TinyMLP_Logit,
    apply_feature_transform,
    apply_robust_minmax,
)


OUT_KIND_X_BOUNDED = 0
OUT_KIND_LINEAR = 1
OUT_KIND_LOG_LINEAR = 2


def _checkpoint_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Inference-bundle loading requires a PyTorch version with torch.load(weights_only=True)."
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected inference-bundle dict in {str(path)!r}, got {type(obj).__name__}.")
    return obj


def decode_output_targets(
    x_raw: torch.Tensor,
    y01: torch.Tensor,
    ft_stats: FeatureTransformStats,
    y_stats: Dict[str, Any],
) -> torch.Tensor:
    q_idx = int(getattr(ft_stats, "q_idx", 0))
    s_idx = int(getattr(ft_stats, "s_idx", 2))
    y_eps = float(y_stats.get("y_eps", 1e-7))
    out_kind = y_stats["out_kind"].to(device=y01.device, dtype=torch.int32)
    out_lo = y_stats["out_lo"].to(device=y01.device, dtype=torch.float32)
    out_invrng = y_stats["out_invrng"].to(device=y01.device, dtype=torch.float32)

    out = torch.empty_like(y01, dtype=torch.float32)
    for idx in range(y01.shape[1]):
        if int(out_kind[idx].item()) == OUT_KIND_X_BOUNDED:
            q = x_raw[:, q_idx : q_idx + 1].to(torch.float32)
            s = x_raw[:, s_idx : s_idx + 1].to(torch.float32)
            x_lo = 1.0 + q - s
            width = torch.clamp(1.0 + q, min=1.0e-12)
            y01c = torch.clamp(y01[:, idx : idx + 1], min=y_eps, max=1.0 - y_eps)
            out[:, idx : idx + 1] = x_lo + y01c * width
        elif int(out_kind[idx].item()) == OUT_KIND_LINEAR:
            out[:, idx : idx + 1] = out_lo[idx] + y01[:, idx : idx + 1] / out_invrng[idx]
        elif int(out_kind[idx].item()) == OUT_KIND_LOG_LINEAR:
            log_out = out_lo[idx] + y01[:, idx : idx + 1] / out_invrng[idx]
            out[:, idx : idx + 1] = torch.exp(log_out)
        else:
            raise ValueError(f"Unsupported output kind {int(out_kind[idx].item())} at index {idx}")
    return out


def relative_error(a: float, b: float) -> float:
    if a != 0.0:
        return float(np.fabs(1.0 - b / a))
    return 1.0 if b != 0.0 else 0.0


def relative_error_array(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true32 = np.asarray(true, dtype=np.float32)
    pred32 = np.asarray(pred, dtype=np.float32)
    out = np.zeros_like(true32)
    nz = true32 != 0.0
    out[nz] = np.abs(1.0 - pred32[nz] / true32[nz])
    out[~nz] = (pred32[~nz] != 0.0).astype(np.float32)
    return out.astype(np.float32, copy=False)


def pick_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict_numpy_like(
    model: nn.Module,
    x: torch.Tensor,
    ft_stats: FeatureTransformStats,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, Any],
) -> torch.Tensor:
    Xt = apply_feature_transform(x, ft_stats)
    mm = {k: x_stats[k] for k in ("lo", "hi", "invrng")}
    X01, _ = apply_robust_minmax(Xt, mm)

    logit = model(X01)
    y01 = torch.sigmoid(logit)
    return decode_output_targets(x, y01, ft_stats, y_stats)


def save_inference_bundle(
    path: str,
    model: torch.nn.Module,
    ft_stats: FeatureTransformStats,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
) -> None:
    base_model = _checkpoint_model(model)
    bundle = {
        "arch": {
            "in_dim": int(base_model.in_dim),
            "hidden_dim": int(base_model.hidden_dim),
            "n_hidden": int(base_model.n_hidden),
            "out_dim": int(base_model.out_dim),
        },
        "state_dict": {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
        "ft_stats": {
            "kind": ft_stats.kind.detach().cpu().to(torch.int32),
            "eps": float(ft_stats.eps),
            "q_idx": int(ft_stats.q_idx),
            "s_idx": int(ft_stats.s_idx),
        },
        "x_stats": {k: v.detach().cpu().to(torch.float32) for k, v in x_stats.items()},
        "y_stats": {k: v.detach().cpu().to(torch.float32) for k, v in y_stats.items()},
    }
    torch.save(bundle, path)
    print(f"Saved inference bundle to: {path}")


def load_inference_bundle(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[
    torch.nn.Module,
    FeatureTransformStats,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    ckpt = _safe_torch_load(path, map_location=map_location)

    arch = ckpt["arch"]
    model = TinyMLP_Logit(
        in_dim=arch["in_dim"],
        hidden_dim=arch["hidden_dim"],
        n_hidden=arch["n_hidden"],
        out_dim=arch["out_dim"],
    ).to(map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ft_d = ckpt["ft_stats"]
    ft_stats = FeatureTransformStats(
        kind=ft_d["kind"].to(torch.int32),
        eps=float(ft_d["eps"]),
        q_idx=int(ft_d.get("q_idx", 0)),
        s_idx=int(ft_d.get("s_idx", 2)),
    )
    x_stats = {k: v.to(torch.float32) for k, v in ckpt["x_stats"].items()}
    y_stats = {k: v.to(torch.float32) for k, v in ckpt["y_stats"].items()}
    return model, ft_stats, x_stats, y_stats


NNPack = Union[
    Tuple[
        nn.Module,
        FeatureTransformStats,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ],
    Dict[str, Any],
]


def _prepare_bundle_on_device(
    neural_net_weights_and_biases: NNPack,
    device: Union[str, torch.device],
) -> Dict[str, Any]:
    if isinstance(neural_net_weights_and_biases, dict):
        model = neural_net_weights_and_biases["model"]
        ft_stats = neural_net_weights_and_biases["ft_stats"]
        x_stats = neural_net_weights_and_biases["x_stats"]
        y_stats = neural_net_weights_and_biases["y_stats"]
    else:
        model, ft_stats, x_stats, y_stats = neural_net_weights_and_biases

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    x_stats_dev = {k: v.to(dev) for k, v in x_stats.items()}
    if hasattr(ft_stats, "kind") and isinstance(ft_stats.kind, torch.Tensor):
        ft_stats.kind = ft_stats.kind.to(dev)

    y_eps_val = y_stats.get("y_eps", torch.tensor(1e-7, dtype=torch.float32))
    y_eps_float = (
        float(y_eps_val.detach().cpu().reshape(-1)[0].item())
        if isinstance(y_eps_val, torch.Tensor)
        else float(y_eps_val)
    )

    width_tiny_val = y_stats.get("width_tiny", torch.tensor(1.0e-12, dtype=torch.float32))
    width_tiny_float = (
        float(width_tiny_val.detach().cpu().reshape(-1)[0].item())
        if isinstance(width_tiny_val, torch.Tensor)
        else float(width_tiny_val)
    )

    y_stats_prepared: Dict[str, Any] = dict(y_stats)
    y_stats_prepared["y_eps"] = y_eps_float
    y_stats_prepared["width_tiny"] = width_tiny_float
    for key in ("out_kind", "out_lo", "out_hi", "out_invrng"):
        if key in y_stats_prepared and isinstance(y_stats_prepared[key], torch.Tensor):
            y_stats_prepared[key] = y_stats_prepared[key].to(dev)

    return {
        "model": model,
        "ft_stats": ft_stats,
        "x_stats": x_stats_dev,
        "y_stats": y_stats_prepared,
        "_scratch_x_in": torch.empty((1, 4), dtype=torch.float32, device=dev),
    }


@torch.inference_mode()
def cons_to_x_guess(
    neural_net_weights_and_biases: NNPack,
    q: float,
    r: float,
    s: float,
    t: float,
    *,
    device: Union[None, str, torch.device] = None,
) -> float:
    if (
        isinstance(neural_net_weights_and_biases, dict)
        and "_scratch_x_in" in neural_net_weights_and_biases
    ):
        bundle = neural_net_weights_and_biases
    else:
        if isinstance(neural_net_weights_and_biases, dict):
            mdev = next(neural_net_weights_and_biases["model"].parameters()).device
        else:
            mdev = next(neural_net_weights_and_biases[0].parameters()).device
        dev = torch.device(device) if device is not None else mdev
        bundle = _prepare_bundle_on_device(neural_net_weights_and_biases, dev)

    model = bundle["model"]
    ft_stats = bundle["ft_stats"]
    x_stats = bundle["x_stats"]
    y_stats = bundle["y_stats"]

    width_tiny = float(y_stats.get("width_tiny", 1.0e-12))
    width = 1.0 + float(q)
    if (not np.isfinite(width)) or (width <= width_tiny):
        raise ValueError(
            f"Invalid Palenzuela width: 1+q={width:.6e} <= width_tiny={width_tiny:.6e}"
        )

    x_in = bundle["_scratch_x_in"]
    x_in[0, 0] = q
    x_in[0, 1] = r
    x_in[0, 2] = s
    x_in[0, 3] = t

    xhat = predict_numpy_like(model, x_in, ft_stats, x_stats, y_stats)
    return float(xhat[0, 0].detach().cpu().item())
