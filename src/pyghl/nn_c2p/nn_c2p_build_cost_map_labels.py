from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


STATE_KIND_ROBUST = 0
STATE_KIND_FAST_ONLY = 1
STATE_KIND_FAILURE_DOMINATED = 2
STATE_KIND_INVALID = 3


def rank_key(success: bool, n_iter: int, cons_error: float) -> tuple[int, int, float]:
    iter_key = int(n_iter) if int(n_iter) >= 0 else int(1e9)
    err_key = float(cons_error) if np.isfinite(cons_error) else float("inf")
    return (0 if success else 1, iter_key, err_key)


def softmax_cost(costs: np.ndarray, temperature: float) -> np.ndarray:
    c = np.asarray(costs, dtype=np.float64)
    cmin = np.nanmin(c)
    logits = -(c - cmin) / max(float(temperature), 1.0e-12)
    logits = np.clip(logits, -100.0, 100.0)
    w = np.exp(logits)
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(c, 1.0 / max(1, c.size))
    return w / total


def build_solver_labels(
    cost_map_file: str | Path,
    *,
    output: str | Path,
    delta_fast: int = 1,
    delta_robust: int = 3,
    window: int = 2,
    f_min: float = 0.6,
    f_fail_max: float = 0.2,
    recovery_error_max: float = 1.0e-6,
    failure_penalty: float = 100.0,
    temperature: float = 1.0,
    gate_margin: float = 1.0,
) -> Path:
    cost_map_file = Path(cost_map_file)
    with h5py.File(cost_map_file, "r") as h5f:
        q = np.asarray(h5f["states/q"][()], dtype=np.float32)
        r = np.asarray(h5f["states/r"][()], dtype=np.float32)
        s = np.asarray(h5f["states/s"][()], dtype=np.float32)
        t = np.asarray(h5f["states/t"][()], dtype=np.float32)
        x_lo = np.asarray(h5f["states/x_lo"][()], dtype=np.float32)
        x_hi = np.asarray(h5f["states/x_hi"][()], dtype=np.float32)
        x_exact = np.asarray(h5f["states/x_exact"][()], dtype=np.float32)
        y_exact = np.asarray(h5f["states/y_exact"][()], dtype=np.float32)
        y_uniform = np.asarray(h5f["grid/y_uniform"][()], dtype=np.float32)

        uniform_success = np.asarray(h5f["uniform/success"][()], dtype=np.bool_)
        uniform_n_iter = np.asarray(h5f["uniform/n_iter"][()], dtype=np.int32)
        uniform_cons_error = np.asarray(h5f["uniform/cons_error"][()], dtype=np.float32)
        baseline_names = [
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in np.asarray(h5f["baselines/names"][()])
        ]
        baseline_success = np.asarray(h5f["baselines/success"][()], dtype=np.bool_)
        baseline_n_iter = np.asarray(h5f["baselines/n_iter"][()], dtype=np.int32)
        baseline_cons_error = np.asarray(h5f["baselines/cons_error"][()], dtype=np.float32)
        baseline_y = np.asarray(h5f["baselines/y"][()], dtype=np.float32)

    N, K = uniform_success.shape
    y_fast_mask = np.zeros((N, K), dtype=np.bool_)
    y_robust_mask = np.zeros((N, K), dtype=np.bool_)
    soft_cost = np.full((N, K), float(failure_penalty), dtype=np.float64)
    soft_prob = np.zeros((N, K), dtype=np.float32)
    state_kind = np.full(N, STATE_KIND_FAILURE_DOMINATED, dtype=np.int32)
    sample_weight = np.zeros(N, dtype=np.float32)
    local_success_fraction = np.zeros((N, K), dtype=np.float32)
    local_failure_fraction = np.zeros((N, K), dtype=np.float32)
    local_median_iter = np.full((N, K), np.inf, dtype=np.float32)
    local_median_cons_error = np.full((N, K), np.inf, dtype=np.float32)
    iter_min_success_arr = np.full(N, np.inf, dtype=np.float64)
    best_baseline_id = np.full(N, -1, dtype=np.int32)
    best_baseline_cost = np.full(N, float(failure_penalty), dtype=np.float64)
    recommended_id = np.full(N, -1, dtype=np.int32)
    best_simple_baseline_id = np.full(N, -1, dtype=np.int32)
    phase_a_beats_simple_baseline = np.zeros(N, dtype=np.int8)
    gate_oracle_use_solver_region = np.zeros(N, dtype=np.int8)

    for i in range(N):
        success_row = uniform_success[i]
        n_iter_row = uniform_n_iter[i]
        cons_err_row = uniform_cons_error[i]

        success_idx = np.flatnonzero(success_row)
        robust_best_cost = float("inf")
        if success_idx.size == 0:
            costs = np.full(K, float(failure_penalty), dtype=np.float64)
            soft_cost[i] = costs
            soft_prob[i] = softmax_cost(costs, temperature).astype(np.float32, copy=False)
            state_kind[i] = STATE_KIND_FAILURE_DOMINATED
            sample_weight[i] = 0.0
        else:
            iter_min_success = int(np.min(n_iter_row[success_row]))
            iter_min_success_arr[i] = float(iter_min_success)
            y_fast_mask[i] = success_row & (n_iter_row <= (iter_min_success + int(delta_fast)))
            state_kind[i] = STATE_KIND_FAST_ONLY
            sample_weight[i] = 0.25
            costs = np.where(success_row, n_iter_row.astype(np.float64), float(failure_penalty))
            soft_cost[i] = costs

            for j in range(K):
                lo = max(0, j - int(window))
                hi = min(K, j + int(window) + 1)
                window_success = success_row[lo:hi]
                window_iters = n_iter_row[lo:hi][window_success]
                window_errs = cons_err_row[lo:hi][window_success]
                local_success_fraction[i, j] = float(window_success.mean())
                local_failure_fraction[i, j] = float((~window_success).mean())
                if window_iters.size > 0:
                    local_median_iter[i, j] = float(np.median(window_iters))
                if window_errs.size > 0:
                    local_median_cons_error[i, j] = float(np.median(window_errs))

            robust = (
                y_fast_mask[i]
                & (local_success_fraction[i] >= float(f_min))
                & (local_failure_fraction[i] <= float(f_fail_max))
                & np.isfinite(local_median_iter[i])
                & (local_median_iter[i] <= float(iter_min_success + int(delta_robust)))
                & np.isfinite(cons_err_row)
                & (cons_err_row <= float(recovery_error_max))
            )

            if not np.any(robust) and np.any(y_fast_mask[i]):
                robust = np.zeros(K, dtype=np.bool_)
                state_kind[i] = STATE_KIND_FAST_ONLY
                sample_weight[i] = 0.25
            else:
                state_kind[i] = STATE_KIND_ROBUST
                sample_weight[i] = 1.0

            y_robust_mask[i] = robust
            if np.any(robust):
                idx = np.flatnonzero(robust)
                soft_prob[i, idx] = 1.0 / float(idx.size)
                robust_best_cost = float(np.min(costs[idx]))
                recommended_id[i] = int(
                    idx[np.argmin(n_iter_row[idx].astype(np.int64))]
                )
            elif np.any(y_fast_mask[i]):
                idx = np.flatnonzero(y_fast_mask[i])
                soft_prob[i, idx] = 1.0 / float(idx.size)
                recommended_id[i] = int(
                    idx[np.argmin(n_iter_row[idx].astype(np.int64))]
                )
            else:
                soft_prob[i] = softmax_cost(costs, temperature).astype(np.float32, copy=False)

        # Best simple baseline uses x_lo, midpoint, x_hi only.
        best_key = None
        best_id = -1
        simple_baseline_ids = [
            idx for idx, name in enumerate(baseline_names) if name in ("x_lo", "midpoint", "x_hi")
        ]
        for baseline_id in simple_baseline_ids:
            key = rank_key(
                bool(baseline_success[i, baseline_id]),
                int(baseline_n_iter[i, baseline_id]),
                float(baseline_cons_error[i, baseline_id]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_id = baseline_id
        best_simple_baseline_id[i] = best_id

        # Best baseline includes phase_a when available.
        best_all_key = None
        best_all_id = -1
        best_all_cost = float(failure_penalty)
        deployable_baseline_ids = [
            idx for idx, name in enumerate(baseline_names) if name != "x_exact"
        ]
        for baseline_id in deployable_baseline_ids:
            if not np.isfinite(baseline_y[i, baseline_id]):
                continue
            key = rank_key(
                bool(baseline_success[i, baseline_id]),
                int(baseline_n_iter[i, baseline_id]),
                float(baseline_cons_error[i, baseline_id]),
            )
            if best_all_key is None or key < best_all_key:
                best_all_key = key
                best_all_id = baseline_id
                best_all_cost = (
                    float(baseline_n_iter[i, baseline_id])
                    if bool(baseline_success[i, baseline_id])
                    else float(failure_penalty)
                )
        best_baseline_id[i] = best_all_id
        best_baseline_cost[i] = best_all_cost
        if np.isfinite(robust_best_cost) and robust_best_cost < (best_all_cost - float(gate_margin)):
            gate_oracle_use_solver_region[i] = 1

        if "phase_a" in baseline_names:
            phase_a_idx = baseline_names.index("phase_a")
        else:
            phase_a_idx = -1
        if phase_a_idx >= 0 and np.isfinite(baseline_y[i, phase_a_idx]):
            phase_key = rank_key(
                bool(baseline_success[i, phase_a_idx]),
                int(baseline_n_iter[i, phase_a_idx]),
                float(baseline_cons_error[i, phase_a_idx]),
            )
            if best_key is not None and phase_key < best_key:
                phase_a_beats_simple_baseline[i] = 1

    output = Path(output)
    with h5py.File(output, "w") as h5f:
        meta = h5f.create_group("meta")
        meta.attrs["source_cost_map_file"] = str(cost_map_file)
        meta.attrs["delta_fast"] = int(delta_fast)
        meta.attrs["delta_robust"] = int(delta_robust)
        meta.attrs["window"] = int(window)
        meta.attrs["f_min"] = float(f_min)
        meta.attrs["f_fail_max"] = float(f_fail_max)
        meta.attrs["recovery_error_max"] = float(recovery_error_max)
        meta.attrs["failure_penalty"] = float(failure_penalty)
        meta.attrs["temperature"] = float(temperature)
        meta.attrs["gate_margin"] = float(gate_margin)
        meta.attrs["baseline_names"] = np.asarray(baseline_names, dtype="S16")

        states = h5f.create_group("states")
        for name, arr in (
            ("q", q),
            ("r", r),
            ("s", s),
            ("t", t),
            ("x_lo", x_lo),
            ("x_hi", x_hi),
            ("x_exact", x_exact),
            ("y_exact", y_exact),
        ):
            states.create_dataset(name, data=arr)

        grid = h5f.create_group("grid")
        grid.create_dataset("y_uniform", data=y_uniform)

        labels = h5f.create_group("labels")
        labels.create_dataset("y_fast_mask", data=y_fast_mask)
        labels.create_dataset("y_robust_mask", data=y_robust_mask)
        labels.create_dataset("soft_cost", data=soft_cost)
        labels.create_dataset("soft_prob", data=soft_prob)
        labels.create_dataset("state_type", data=state_kind)
        labels.create_dataset("state_kind", data=state_kind)
        labels.create_dataset("sample_weight", data=sample_weight)
        labels.create_dataset("iter_min_success", data=iter_min_success_arr)
        labels.create_dataset("local_success_fraction", data=local_success_fraction)
        labels.create_dataset("local_failure_fraction", data=local_failure_fraction)
        labels.create_dataset("local_median_iter", data=local_median_iter)
        labels.create_dataset("local_median_cons_error", data=local_median_cons_error)
        labels.create_dataset("best_baseline_id", data=best_baseline_id)
        labels.create_dataset("best_baseline_cost", data=best_baseline_cost)
        labels.create_dataset("best_simple_baseline_id", data=best_simple_baseline_id)
        labels.create_dataset("recommended_id", data=recommended_id)
        labels.create_dataset(
            "phase_a_beats_simple_baseline", data=phase_a_beats_simple_baseline
        )
        labels.create_dataset(
            "gate_oracle_use_solver_region", data=gate_oracle_use_solver_region
        )

    print(f"Wrote solver labels to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cost_map_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta_fast", type=int, default=1)
    parser.add_argument("--delta_robust", type=int, default=3)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--f_min", type=float, default=0.6)
    parser.add_argument("--f_fail_max", type=float, default=0.2)
    parser.add_argument("--recovery_error_max", type=float, default=1.0e-6)
    parser.add_argument("--failure_penalty", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gate_margin", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_solver_labels(
        args.cost_map_file,
        output=args.output,
        delta_fast=args.delta_fast,
        delta_robust=args.delta_robust,
        window=args.window,
        f_min=args.f_min,
        f_fail_max=args.f_fail_max,
        recovery_error_max=args.recovery_error_max,
        failure_penalty=args.failure_penalty,
        temperature=args.temperature,
        gate_margin=args.gate_margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
