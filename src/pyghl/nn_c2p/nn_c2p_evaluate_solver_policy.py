from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import pyghl as ghl

from .._nn_common import apply_feature_transform, apply_robust_minmax
from .common import set_flat_metric
from .nn_c2p_generate_cost_maps import make_eos, make_params
from .nn_c2p_generate_dataset import compute_cons_error, reconstruct_prims_from_x


def _policy_metrics(success: np.ndarray, n_iter: np.ndarray, cons_err: np.ndarray, *, failure_penalty: float) -> dict[str, float]:
    success = np.asarray(success, dtype=bool)
    n_iter = np.asarray(n_iter, dtype=np.int32)
    cons_err = np.asarray(cons_err, dtype=np.float64)
    penalized = np.where(success, n_iter.astype(np.float64), float(failure_penalty))
    success_iters = n_iter[success].astype(np.float64)
    out = {
        "failure_rate": float(np.mean(~success)),
        "success_rate": float(np.mean(success)),
        "mean_penalized_cost": float(np.mean(penalized)),
        "mean_iter_success": float(np.mean(success_iters)) if success_iters.size else float("nan"),
        "median_iter_success": float(np.median(success_iters)) if success_iters.size else float("nan"),
        "p90_iter_success": float(np.percentile(success_iters, 90)) if success_iters.size else float("nan"),
        "p95_iter_success": float(np.percentile(success_iters, 95)) if success_iters.size else float("nan"),
        "p99_iter_success": float(np.percentile(success_iters, 99)) if success_iters.size else float("nan"),
        "max_iter_success": float(np.max(success_iters)) if success_iters.size else float("nan"),
        "mean_cons_err_success": float(np.nanmean(cons_err[success])) if success.any() else float("nan"),
    }
    out["score"] = (
        100.0 * out["failure_rate"]
        + (0.0 if np.isnan(out["mean_iter_success"]) else out["mean_iter_success"])
        + 0.25 * (0.0 if np.isnan(out["p95_iter_success"]) else out["p95_iter_success"])
        + 0.10 * (0.0 if np.isnan(out["p99_iter_success"]) else out["p99_iter_success"])
    )
    return out


def _load_data(cost_map_file: str | Path, label_file: str | Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(cost_map_file, "r") as h5f:
        out["table"] = str(h5f["meta"].attrs["table"])
        out["q"] = np.asarray(h5f["states/q"][()], dtype=np.float32)
        out["r"] = np.asarray(h5f["states/r"][()], dtype=np.float32)
        out["s"] = np.asarray(h5f["states/s"][()], dtype=np.float32)
        out["t"] = np.asarray(h5f["states/t"][()], dtype=np.float32)
        out["rho"] = np.asarray(h5f["states/rho"][()], dtype=np.float32)
        out["T"] = np.asarray(h5f["states/T"][()], dtype=np.float32)
        out["Ye"] = np.asarray(h5f["states/Ye"][()], dtype=np.float32)
        out["vx"] = np.asarray(h5f["states/vx"][()], dtype=np.float32)
        out["vy"] = np.asarray(h5f["states/vy"][()], dtype=np.float32)
        out["vz"] = np.asarray(h5f["states/vz"][()], dtype=np.float32)
        out["Bx"] = np.asarray(h5f["states/Bx"][()], dtype=np.float32)
        out["By"] = np.asarray(h5f["states/By"][()], dtype=np.float32)
        out["Bz"] = np.asarray(h5f["states/Bz"][()], dtype=np.float32)
        out["y_uniform"] = np.asarray(h5f["grid/y_uniform"][()], dtype=np.float32)
        out["uniform_success"] = np.asarray(h5f["uniform/success"][()], dtype=np.bool_)
        out["uniform_n_iter"] = np.asarray(h5f["uniform/n_iter"][()], dtype=np.int32)
        out["uniform_cons_err"] = np.asarray(h5f["uniform/cons_error"][()], dtype=np.float64)
        out["baseline_names"] = np.asarray(h5f["baselines/names"][()])
        out["baseline_x"] = np.asarray(h5f["baselines/x"][()], dtype=np.float32)
        out["baseline_success"] = np.asarray(h5f["baselines/success"][()], dtype=np.bool_)
        out["baseline_n_iter"] = np.asarray(h5f["baselines/n_iter"][()], dtype=np.int32)
        out["baseline_cons_err"] = np.asarray(h5f["baselines/cons_error"][()], dtype=np.float64)
    with h5py.File(label_file, "r") as h5f:
        out["recommended_id"] = np.asarray(h5f["labels/recommended_id"][()], dtype=np.int32)
        out["best_baseline_id"] = np.asarray(h5f["labels/best_baseline_id"][()], dtype=np.int32)
        out["state_type"] = np.asarray(h5f["labels/state_type"][()], dtype=np.int32)
    out["X"] = np.stack((out["q"], out["r"], out["s"], out["t"]), axis=1).astype(np.float32)
    return out


def _bundle_forward(bundle_path: str | Path, X: np.ndarray) -> tuple[torch.nn.Module, object, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    model, ft_stats, x_stats, y_stats = ghl.nn.load_inference_bundle(str(bundle_path))
    X_t = torch.tensor(X, dtype=torch.float32)
    Xt = apply_feature_transform(X_t, ft_stats)
    X01, _ = apply_robust_minmax(Xt, {k: x_stats[k] for k in ("lo", "hi", "invrng")})
    logits = model(X01).to(torch.float32)
    return model, ft_stats, x_stats, y_stats, logits


def _build_runtime(data: dict[str, np.ndarray]):
    params = make_params()
    eos = make_eos(params, data["table"])
    metric, metric_aux = set_flat_metric()
    return params, eos, metric, metric_aux


def _build_reference_state(
    data: dict[str, np.ndarray],
    runtime,
    i: int,
):
    params, eos, metric, metric_aux = runtime
    rho = float(data["rho"][i])
    temp = float(data["T"][i])
    ye = float(data["Ye"][i])
    prs, eps, ent = eos.tabulated_compute_P_eps_S_from_T(rho, ye, temp)
    prims = ghl.initialize_primitives(
        rho,
        prs,
        eps,
        float(data["vx"][i]),
        float(data["vy"][i]),
        float(data["vz"][i]),
        float(data["Bx"][i]),
        float(data["By"][i]),
        float(data["Bz"][i]),
        ent,
        ye,
        temp,
    )
    ghl.limit_v_and_compute_u0(params, metric, prims)
    cons = ghl.compute_conservs(metric, metric_aux, prims)
    cons_undens = ghl.undensitize_conservatives(metric.sqrt_detgamma, cons)
    return prims, cons_undens


def _nearest_uniform_eval(data: dict[str, np.ndarray], y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_uniform = data["y_uniform"]
    idx = np.argmin(np.abs(y_uniform[None, :] - y_pred[:, None]), axis=1)
    rows = np.arange(y_pred.shape[0], dtype=np.int64)
    return (
        data["uniform_success"][rows, idx],
        data["uniform_n_iter"][rows, idx],
        data["uniform_cons_err"][rows, idx],
    )


def _evaluate_actual_x_predictions(
    data: dict[str, np.ndarray],
    runtime,
    x_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = x_pred.shape[0]
    success = np.zeros(N, dtype=bool)
    n_iter = np.full(N, -1, dtype=np.int32)
    cons_err = np.full(N, np.nan, dtype=np.float64)
    params, eos, metric, metric_aux = runtime
    for i in range(N):
        prims_ref, cons_undens = _build_reference_state(data, runtime, i)
        B_fields = (float(data["Bx"][i]), float(data["By"][i]), float(data["Bz"][i]))
        try:
            prims_guess = reconstruct_prims_from_x(
                params, eos, metric, cons_undens, B_fields, float(x_pred[i])
            )
        except Exception:
            continue
        diagnostics = ghl.initialize_diagnostics()
        try:
            ghl.tabulated_con2prim_multi_method(
                params, eos, metric, metric_aux, cons_undens, prims_guess, diagnostics
            )
            cons_recovered = ghl.compute_conservs(metric, metric_aux, prims_guess)
            cons_recovered_undens = ghl.undensitize_conservatives(
                metric.sqrt_detgamma, cons_recovered
            )
            success[i] = True
            n_iter[i] = int(diagnostics.n_iter)
            cons_err[i] = float(compute_cons_error(cons_undens, cons_recovered_undens))
        except ghl.GRHayLError:
            n_iter[i] = int(getattr(diagnostics, "n_iter", -1))
    return success, n_iter, cons_err


def _evaluate_bundle_region(data: dict[str, np.ndarray], runtime, bundle_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, _, logits = _bundle_forward(bundle_path, data["X"])
    y_pred = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    x_lo = 1.0 + data["q"] - data["s"]
    width = 1.0 + data["q"]
    x_pred = x_lo + y_pred * width
    return _evaluate_actual_x_predictions(data, runtime, x_pred)


def _evaluate_bundle_soft_bins(data: dict[str, np.ndarray], runtime, bundle_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, _, logits = _bundle_forward(bundle_path, data["X"])
    idx = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    y_pred = data["y_uniform"][idx]
    x_lo = 1.0 + data["q"] - data["s"]
    width = 1.0 + data["q"]
    x_pred = x_lo + y_pred * width
    return _evaluate_actual_x_predictions(data, runtime, x_pred)


def _evaluate_gate_policy(
    data: dict[str, np.ndarray],
    runtime,
    *,
    region_bundle: str | Path,
    gate_bundle: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, _, gate_logits = _bundle_forward(gate_bundle, data["X"])
    gate = (torch.sigmoid(gate_logits).detach().cpu().numpy().reshape(-1) >= 0.5)
    region_success, region_iter, region_err = _evaluate_bundle_region(data, runtime, region_bundle)

    base_success, base_iter, base_err = _evaluate_actual_x_predictions(
        data,
        runtime,
        np.asarray(
            [
                data["baseline_x"][i, data["best_baseline_id"][i]]
                if data["best_baseline_id"][i] >= 0
                else np.nan
                for i in range(data["X"].shape[0])
            ],
            dtype=np.float32,
        ),
    )

    success = np.where(gate, region_success, base_success)
    n_iter = np.where(gate, region_iter, base_iter)
    cons_err = np.where(gate, region_err, base_err)
    return success, n_iter, cons_err


def evaluate(
    cost_map_file: str | Path,
    label_file: str | Path,
    *,
    output: str | Path | None = None,
    failure_penalty: float = 100.0,
    region_bundle: str | Path | None = None,
    softbin_bundle: str | Path | None = None,
    gate_bundle: str | Path | None = None,
) -> str:
    data = _load_data(cost_map_file, label_file)
    runtime = _build_runtime(data)
    baseline_names = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in data["baseline_names"]
    ]

    lines = [
        f"cost_map_file: {cost_map_file}",
        f"label_file: {label_file}",
        f"failure_penalty: {failure_penalty:.6g}",
    ]

    for baseline_id, name in enumerate(baseline_names):
        metrics = _policy_metrics(
            data["baseline_success"][:, baseline_id],
            data["baseline_n_iter"][:, baseline_id],
            data["baseline_cons_err"][:, baseline_id],
            failure_penalty=failure_penalty,
        )
        lines.append(
            f"{name}: failure_rate={metrics['failure_rate']:.6f} "
            f"mean_iter_success={metrics['mean_iter_success']:.6f} "
            f"p95_iter_success={metrics['p95_iter_success']:.6f} "
            f"score={metrics['score']:.6f}"
        )

    oracle_ok = data["recommended_id"] >= 0
    rec_success = np.zeros_like(oracle_ok, dtype=bool)
    rec_iter = np.full_like(data["recommended_id"], -1, dtype=np.int32)
    rec_err = np.full(data["recommended_id"].shape, np.nan, dtype=np.float64)
    if np.any(oracle_ok):
        idx = data["recommended_id"][oracle_ok]
        rows = np.flatnonzero(oracle_ok)
        rec_success[rows] = data["uniform_success"][rows, idx]
        rec_iter[rows] = data["uniform_n_iter"][rows, idx]
        rec_err[rows] = data["uniform_cons_err"][rows, idx]
    rec_metrics = _policy_metrics(rec_success, rec_iter, rec_err, failure_penalty=failure_penalty)
    lines.append(
        f"best_robust_y_oracle: failure_rate={rec_metrics['failure_rate']:.6f} "
        f"mean_iter_success={rec_metrics['mean_iter_success']:.6f} "
        f"p95_iter_success={rec_metrics['p95_iter_success']:.6f} "
        f"score={rec_metrics['score']:.6f}"
    )

    best_sampled_success = np.zeros(data["uniform_success"].shape[0], dtype=bool)
    best_sampled_iter = np.full(data["uniform_success"].shape[0], -1, dtype=np.int32)
    best_sampled_err = np.full(data["uniform_success"].shape[0], np.nan, dtype=np.float64)
    for i in range(data["uniform_success"].shape[0]):
        key_best = None
        j_best = -1
        for j in range(data["uniform_success"].shape[1]):
            key = (
                0 if bool(data["uniform_success"][i, j]) else 1,
                int(data["uniform_n_iter"][i, j]) if int(data["uniform_n_iter"][i, j]) >= 0 else int(1e9),
                float(data["uniform_cons_err"][i, j]) if np.isfinite(data["uniform_cons_err"][i, j]) else float("inf"),
            )
            if key_best is None or key < key_best:
                key_best = key
                j_best = j
        if j_best >= 0:
            best_sampled_success[i] = data["uniform_success"][i, j_best]
            best_sampled_iter[i] = data["uniform_n_iter"][i, j_best]
            best_sampled_err[i] = data["uniform_cons_err"][i, j_best]
    sampled_metrics = _policy_metrics(best_sampled_success, best_sampled_iter, best_sampled_err, failure_penalty=failure_penalty)
    lines.append(
        f"best_sampled_y_oracle: failure_rate={sampled_metrics['failure_rate']:.6f} "
        f"mean_iter_success={sampled_metrics['mean_iter_success']:.6f} "
        f"p95_iter_success={sampled_metrics['p95_iter_success']:.6f} "
        f"score={sampled_metrics['score']:.6f}"
    )

    valid = data["best_baseline_id"] >= 0
    rows = np.flatnonzero(valid)
    cols = data["best_baseline_id"][valid]
    selector_success = np.zeros(data["X"].shape[0], dtype=bool)
    selector_iter = np.full(data["X"].shape[0], -1, dtype=np.int32)
    selector_err = np.full(data["X"].shape[0], np.nan, dtype=np.float64)
    selector_success[rows] = data["baseline_success"][rows, cols]
    selector_iter[rows] = data["baseline_n_iter"][rows, cols]
    selector_err[rows] = data["baseline_cons_err"][rows, cols]
    selector_metrics = _policy_metrics(selector_success, selector_iter, selector_err, failure_penalty=failure_penalty)
    lines.append(
        f"best_baseline_selector: failure_rate={selector_metrics['failure_rate']:.6f} "
        f"mean_iter_success={selector_metrics['mean_iter_success']:.6f} "
        f"p95_iter_success={selector_metrics['p95_iter_success']:.6f} "
        f"score={selector_metrics['score']:.6f}"
    )

    if region_bundle is not None:
        success, n_iter, cons_err = _evaluate_bundle_region(data, runtime, region_bundle)
        metrics = _policy_metrics(success, n_iter, cons_err, failure_penalty=failure_penalty)
        lines.append(
            f"solver_region_bundle: failure_rate={metrics['failure_rate']:.6f} "
            f"mean_iter_success={metrics['mean_iter_success']:.6f} "
            f"p95_iter_success={metrics['p95_iter_success']:.6f} "
            f"score={metrics['score']:.6f}"
        )

    if softbin_bundle is not None:
        success, n_iter, cons_err = _evaluate_bundle_soft_bins(data, runtime, softbin_bundle)
        metrics = _policy_metrics(success, n_iter, cons_err, failure_penalty=failure_penalty)
        lines.append(
            f"solver_softbin_bundle: failure_rate={metrics['failure_rate']:.6f} "
            f"mean_iter_success={metrics['mean_iter_success']:.6f} "
            f"p95_iter_success={metrics['p95_iter_success']:.6f} "
            f"score={metrics['score']:.6f}"
        )

    if region_bundle is not None and gate_bundle is not None:
        success, n_iter, cons_err = _evaluate_gate_policy(
            data, runtime, region_bundle=region_bundle, gate_bundle=gate_bundle
        )
        metrics = _policy_metrics(success, n_iter, cons_err, failure_penalty=failure_penalty)
        lines.append(
            f"solver_gate_policy: failure_rate={metrics['failure_rate']:.6f} "
            f"mean_iter_success={metrics['mean_iter_success']:.6f} "
            f"p95_iter_success={metrics['p95_iter_success']:.6f} "
            f"score={metrics['score']:.6f}"
        )

    text = "\n".join(lines) + "\n"
    if output is not None:
        Path(output).write_text(text, encoding="utf-8")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cost_map_file", type=Path)
    parser.add_argument("label_file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure_penalty", type=float, default=100.0)
    parser.add_argument("--region_bundle", type=Path)
    parser.add_argument("--softbin_bundle", type=Path)
    parser.add_argument("--gate_bundle", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    text = evaluate(
        args.cost_map_file,
        args.label_file,
        output=args.output,
        failure_penalty=args.failure_penalty,
        region_bundle=args.region_bundle,
        softbin_bundle=args.softbin_bundle,
        gate_bundle=args.gate_bundle,
    )
    if args.output is None:
        print(text, end="")
    else:
        print(f"Wrote evaluator summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
