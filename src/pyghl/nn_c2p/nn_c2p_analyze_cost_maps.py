from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from .nn_c2p_build_cost_map_labels import (
    STATE_KIND_FAILURE_DOMINATED,
    STATE_KIND_FAST_ONLY,
    STATE_KIND_INVALID,
    STATE_KIND_ROBUST,
)


def count_components(mask: np.ndarray) -> int:
    count = 0
    inside = False
    for value in np.asarray(mask, dtype=bool):
        if value and not inside:
            count += 1
            inside = True
        elif not value:
            inside = False
    return count


def _load(cost_map_file: str | Path, label_file: str | Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(cost_map_file, "r") as h5f:
        out["y_uniform"] = np.asarray(h5f["grid/y_uniform"][()], dtype=np.float32)
        out["q"] = np.asarray(h5f["states/q"][()], dtype=np.float32)
        out["r"] = np.asarray(h5f["states/r"][()], dtype=np.float32)
        out["s"] = np.asarray(h5f["states/s"][()], dtype=np.float32)
        out["t"] = np.asarray(h5f["states/t"][()], dtype=np.float32)
        out["uniform_success"] = np.asarray(h5f["uniform/success"][()], dtype=np.bool_)
        out["uniform_n_iter"] = np.asarray(h5f["uniform/n_iter"][()], dtype=np.int32)
        out["baseline_names"] = np.asarray(h5f["baselines/names"][()])
        out["baseline_y"] = np.asarray(h5f["baselines/y"][()], dtype=np.float32)
    with h5py.File(label_file, "r") as h5f:
        out["y_fast_mask"] = np.asarray(h5f["labels/y_fast_mask"][()], dtype=np.bool_)
        out["y_robust_mask"] = np.asarray(h5f["labels/y_robust_mask"][()], dtype=np.bool_)
        out["soft_cost"] = np.asarray(h5f["labels/soft_cost"][()], dtype=np.float64)
        out["soft_prob"] = np.asarray(h5f["labels/soft_prob"][()], dtype=np.float32)
        out["state_type"] = np.asarray(h5f["labels/state_type"][()], dtype=np.int32)
        out["best_baseline_id"] = np.asarray(
            h5f["labels/best_baseline_id"][()], dtype=np.int32
        )
        out["recommended_id"] = np.asarray(
            h5f["labels/recommended_id"][()], dtype=np.int32
        )
        out["phase_a_beats_simple_baseline"] = np.asarray(
            h5f["labels/phase_a_beats_simple_baseline"][()], dtype=np.int8
        )
    return out


def _summary_lines(cost_map_file: Path, label_file: Path, data: dict[str, np.ndarray]) -> list[str]:
    y_fast_mask = data["y_fast_mask"]
    y_robust_mask = data["y_robust_mask"]
    state_type = data["state_type"]
    best_baseline_id = data["best_baseline_id"]
    phase_a_beats = data["phase_a_beats_simple_baseline"]
    recommended_id = data["recommended_id"]
    y_uniform = data["y_uniform"]
    baseline_y = data["baseline_y"]
    baseline_names = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in data["baseline_names"]
    ]

    N, _ = y_fast_mask.shape
    fast_width = y_fast_mask.sum(axis=1)
    robust_width = y_robust_mask.sum(axis=1)
    robust_components = np.asarray([count_components(row) for row in y_robust_mask], dtype=np.int32)
    best_y = np.where(
        recommended_id >= 0,
        y_uniform[np.clip(recommended_id, 0, y_uniform.size - 1)],
        np.nan,
    )
    phase_a_in_robust = np.zeros(N, dtype=bool)
    if baseline_y.shape[1] >= 4:
        phase_a_valid = np.isfinite(baseline_y[:, 3])
        phase_a_idx = np.clip(
            np.argmin(np.abs(y_uniform[None, :] - baseline_y[:, 3:4]), axis=1),
            0,
            y_uniform.size - 1,
        )
        phase_a_in_robust[phase_a_valid] = y_robust_mask[phase_a_valid, phase_a_idx[phase_a_valid]]

    lines = [
        f"cost_map_file: {cost_map_file}",
        f"label_file: {label_file}",
        f"n_states: {N}",
        (
            "state_type fractions: "
            f"robust={np.mean(state_type == STATE_KIND_ROBUST):.6f} "
            f"fast_only={np.mean(state_type == STATE_KIND_FAST_ONLY):.6f} "
            f"all_failed={np.mean(state_type == STATE_KIND_FAILURE_DOMINATED):.6f} "
            f"invalid={np.mean(state_type == STATE_KIND_INVALID):.6f}"
        ),
        (
            "best_baseline fractions: "
            + " ".join(
                f"{name}={np.mean(best_baseline_id == idx):.6f}"
                for idx, name in enumerate(baseline_names)
                if name != "x_exact"
            )
        ),
        f"phase_a_in_Y_robust fraction: {np.mean(phase_a_in_robust):.6f}",
        f"phase_a_beats_simple_baseline fraction: {np.mean(phase_a_beats != 0):.6f}",
        (
            f"Y_fast width: mean={fast_width.mean():.3f} median={np.median(fast_width):.3f} "
            f"p95={np.percentile(fast_width,95):.3f} p99={np.percentile(fast_width,99):.3f}"
        ),
        (
            f"Y_robust width: mean={robust_width.mean():.3f} median={np.median(robust_width):.3f} "
            f"p95={np.percentile(robust_width,95):.3f} p99={np.percentile(robust_width,99):.3f}"
        ),
        (
            "robust connected components: "
            f"mean={robust_components.mean():.3f} median={np.median(robust_components):.3f} "
            f"max={robust_components.max()}"
        ),
        (
            "robust basin type fractions: "
            f"single_contiguous={np.mean((robust_width > 0) & (robust_components == 1)):.6f} "
            f"multimodal={np.mean(robust_components >= 2):.6f} "
            f"empty={np.mean(robust_width == 0):.6f}"
        ),
        (
            f"recommended_y: mean={np.nanmean(best_y):.6f} "
            f"median={np.nanmedian(best_y):.6f}"
        ),
    ]
    return lines


def _maybe_import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_hist(prefix: Path, name: str, values: np.ndarray, *, bins=30, xlabel: str = "", title: str = ""):
    plt = _maybe_import_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + name))
    plt.close(fig)


def _plot_bar(prefix: Path, name: str, labels: list[str], values: np.ndarray, *, title: str = ""):
    plt = _maybe_import_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + name))
    plt.close(fig)


def _plot_example_maps(prefix: Path, name: str, indices: np.ndarray, data: dict[str, np.ndarray], title: str):
    if indices.size == 0:
        return
    plt = _maybe_import_matplotlib()
    n_show = min(3, int(indices.size))
    fig, axes = plt.subplots(n_show, 1, figsize=(7, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    y = data["y_uniform"]
    for ax, idx in zip(axes, indices[:n_show]):
        cost = data["soft_cost"][idx]
        ax.plot(y, cost, color="black", lw=1.5, label="soft_cost")
        ax.scatter(y[data["y_fast_mask"][idx]], cost[data["y_fast_mask"][idx]], s=18, label="Y_fast")
        ax.scatter(
            y[data["y_robust_mask"][idx]],
            cost[data["y_robust_mask"][idx]],
            s=22,
            label="Y_robust",
        )
        ax.set_ylabel(f"state {int(idx)}")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("y")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + name))
    plt.close(fig)


def _plot_qrs_failure(prefix: Path, data: dict[str, np.ndarray]):
    plt = _maybe_import_matplotlib()
    q = data["q"]
    r = data["r"]
    s = data["s"]
    state_type = data["state_type"]
    failed = (state_type == STATE_KIND_FAILURE_DOMINATED).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    pairs = [(q, r, "q", "r"), (q, s, "q", "s"), (r, s, "r", "s")]
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        hb = ax.hexbin(x, y, C=failed, reduce_C_function=np.mean, gridsize=25, mincnt=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(hb, ax=ax, label="failure rate")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_qrs_bins_failure_rate.png"))
    plt.close(fig)


def analyze(cost_map_file: str | Path, label_file: str | Path, *, prefix: str | Path) -> Path:
    cost_map_file = Path(cost_map_file)
    label_file = Path(label_file)
    prefix = Path(prefix)
    data = _load(cost_map_file, label_file)

    lines = _summary_lines(cost_map_file, label_file, data)
    summary_path = prefix.with_name(prefix.name + "_summary.txt")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    robust_width = data["y_robust_mask"].sum(axis=1)
    state_type = data["state_type"]
    recommended_id = data["recommended_id"]
    best_y = data["y_uniform"][np.clip(recommended_id, 0, data["y_uniform"].size - 1)]
    best_y = best_y[recommended_id >= 0]
    best_baseline_id = data["best_baseline_id"]
    baseline_name_pairs = [
        (
            i,
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x),
        )
        for i, x in enumerate(data["baseline_names"])
    ]
    baseline_name_pairs = [(i, name) for i, name in baseline_name_pairs if name != "x_exact"]
    baseline_names = [name for _, name in baseline_name_pairs]
    baseline_fracs = np.asarray(
        [np.mean(best_baseline_id == i) for i, _ in baseline_name_pairs], dtype=np.float64
    )

    _plot_hist(prefix, "_robust_width_hist.png", robust_width, bins=30, xlabel="|Y_robust|", title="Robust Basin Width")
    _plot_bar(
        prefix,
        "_state_type_hist.png",
        ["robust", "fast_only", "all_failed", "invalid"],
        np.asarray(
            [
                np.mean(state_type == STATE_KIND_ROBUST),
                np.mean(state_type == STATE_KIND_FAST_ONLY),
                np.mean(state_type == STATE_KIND_FAILURE_DOMINATED),
                np.mean(state_type == STATE_KIND_INVALID),
            ],
            dtype=np.float64,
        ),
        title="State Type Fractions",
    )
    if best_y.size > 0:
        _plot_hist(prefix, "_best_y_hist.png", best_y, bins=30, xlabel="recommended y", title="Recommended y")
    _plot_bar(prefix, "_baseline_win_rates.png", baseline_names, baseline_fracs, title="Baseline Win Rates")

    robust_components = np.asarray(
        [count_components(row) for row in data["y_robust_mask"]], dtype=np.int32
    )
    single = np.flatnonzero((robust_width > 0) & (robust_components == 1))
    multimodal = np.flatnonzero(robust_components >= 2)
    cliff = np.flatnonzero((robust_width == 0) & (data["y_fast_mask"].sum(axis=1) > 0))
    _plot_example_maps(prefix, "_map_examples_single_basin.png", single, data, "Single Robust Basin")
    _plot_example_maps(prefix, "_map_examples_multimodal.png", multimodal, data, "Multimodal Robust Basin")
    _plot_example_maps(prefix, "_map_examples_cliff.png", cliff, data, "Cliff / Fast-only Maps")
    _plot_qrs_failure(prefix, data)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cost_map_file", type=Path)
    parser.add_argument("label_file", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path = analyze(args.cost_map_file, args.label_file, prefix=args.prefix)
    print(f"Wrote analysis summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
