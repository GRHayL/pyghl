#!/usr/bin/env python3
"""Plot the EOS specific internal energy as a function of temperature."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

MEV_TO_ERG = 1.602176634e-6
M_BARYON_G = 1.66053906660e-24
MEV_PER_BARYON_TO_ERG_PER_G = MEV_TO_ERG / M_BARYON_G


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot specific internal energy versus temperature from one or more "
            "tabulated EOS HDF5 files."
        )
    )
    parser.add_argument(
        "eos_files",
        nargs="+",
        help="Input EOS HDF5 files, e.g. sly4.h5 dd2.h5",
    )
    parser.add_argument(
        "--rho-index",
        type=int,
        default=None,
        help="Use this rho index instead of auto-selecting a degenerate slice.",
    )
    parser.add_argument(
        "--ye-index",
        type=int,
        default=None,
        help="Use this Y_e index instead of auto-selecting a degenerate slice.",
    )
    parser.add_argument(
        "--slope-tol",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed change in log10(eps+eps0) between adjacent "
            "temperature points while identifying the low-temperature plateau."
        ),
    )
    parser.add_argument(
        "--min-rise",
        type=float,
        default=1.0,
        help=(
            "Minimum total rise in log10(eps+eps0) across temperature required "
            "for the auto-selected slice."
        ),
    )
    parser.add_argument(
        "--physical-axes",
        action="store_true",
        help=(
            "Plot physical T and eps values and use logarithmic axis scales. "
            "By default the script plots the tabulated log10 quantities directly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the figure to this file instead of opening a window.",
    )
    parser.add_argument(
        "--draw-connector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlay a smooth connector that replaces the degenerate plateau by "
            "a small positive slope in the plotted variables."
        ),
    )
    parser.add_argument(
        "--connector-slope",
        type=float,
        default=1.0e-1,
        help=(
            "Target dy/dx for the low-temperature extrapolation in plotted log "
            "coordinates. The script scans backward from T_max and starts the "
            "extrapolation where the table slope matches this value."
        ),
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=None,
        help=(
            "Write a copy of the input EOS HDF5 file with logenergy overwritten "
            "for every (Ye, rho) slice using the connector construction."
        ),
    )
    return parser.parse_args()


def choose_slice(
    logenergy: np.ndarray,
    slope_tol: float,
    min_rise: float,
) -> tuple[int, int]:
    """Choose a (Ye, rho) slice with a flat low-T plateau and later rise."""
    n_ye, _, n_rho = logenergy.shape
    best_score: tuple[int, float] | None = None
    best_indices: tuple[int, int] | None = None

    for iy in range(n_ye):
        for ir in range(n_rho):
            curve = logenergy[iy, :, ir]
            if not np.all(np.isfinite(curve)):
                continue

            total_rise = float(curve[-1] - curve[0])
            if total_rise < min_rise:
                continue

            diffs = np.diff(curve)
            plateau_len = 1
            for diff in diffs:
                if abs(diff) <= slope_tol:
                    plateau_len += 1
                else:
                    break

            score = (plateau_len, total_rise)
            if best_score is None or score > best_score:
                best_score = score
                best_indices = (iy, ir)

    if best_indices is None:
        raise ValueError(
            "Could not find a slice with a sufficiently degenerate low-temperature region. "
            "Try reducing --min-rise or increasing --slope-tol."
        )
    return best_indices


def label_from_path(path: Path) -> str:
    return path.stem.upper()


def rewrite_logenergy_table(
    input_path: Path,
    output_path: Path,
    connector_slope: float,
) -> tuple[int, int, int, int]:
    """Copy an EOS table and overwrite logenergy/logpress using the connector."""
    if output_path.resolve() == input_path.resolve():
        raise ValueError("--output-table must be different from the input EOS file")

    output_path.write_bytes(input_path.read_bytes())

    changed_slices = 0
    total_slices = 0
    changed_points = 0
    pressure_updated_points = 0
    with h5py.File(output_path, "r+") as f:
        logtemp = np.asarray(f["logtemp"][:], dtype=float)
        logenergy = np.asarray(f["logenergy"][:], dtype=float)
        logpress = np.asarray(f["logpress"][:], dtype=float)
        entropy = np.asarray(f["entropy"][:], dtype=float)
        logrho = np.asarray(f["logrho"][:], dtype=float)
        energy_shift = float(np.asarray(f["energy_shift"][:]).reshape(-1)[0])
        updated = logenergy.copy()
        updated_logpress = logpress.copy()

        for iy in range(logenergy.shape[0]):
            for ir in range(logenergy.shape[2]):
                total_slices += 1
                result = build_connector_curve(
                    logtemp,
                    logenergy[iy, :, ir],
                    connector_slope,
                )
                if result is None:
                    continue
                connector_curve, _ = result
                changed_mask = np.abs(connector_curve - logenergy[iy, :, ir]) > 1e-14
                if np.any(changed_mask):
                    changed_slices += 1
                    changed_points += int(np.count_nonzero(changed_mask))
                updated[iy, :, ir] = connector_curve

        rho = np.power(10.0, logrho)
        T_mev = np.power(10.0, logtemp)
        eps = np.power(10.0, updated) - energy_shift
        Ts = entropy * T_mev[None, :, None] * MEV_PER_BARYON_TO_ERG_PER_G
        free_energy = eps - Ts
        dadrho = np.empty_like(free_energy)

        for ir in range(len(rho)):
            if ir == 0:
                dadrho[:, :, ir] = (free_energy[:, :, ir + 1] - free_energy[:, :, ir]) / (
                    rho[ir + 1] - rho[ir]
                )
            elif ir == len(rho) - 1:
                dadrho[:, :, ir] = (free_energy[:, :, ir] - free_energy[:, :, ir - 1]) / (
                    rho[ir] - rho[ir - 1]
                )
            else:
                dadrho[:, :, ir] = (
                    free_energy[:, :, ir + 1] - free_energy[:, :, ir - 1]
                ) / (rho[ir + 1] - rho[ir - 1])

        reconstructed_pressure = rho[None, None, :] ** 2 * dadrho
        changed_mask_all = np.abs(updated - logenergy) > 1e-14
        valid_pressure_mask = changed_mask_all & np.isfinite(reconstructed_pressure) & (
            reconstructed_pressure > 0.0
        )
        if np.any(valid_pressure_mask):
            updated_logpress[valid_pressure_mask] = np.log10(
                reconstructed_pressure[valid_pressure_mask]
            )
            pressure_updated_points = int(np.count_nonzero(valid_pressure_mask))

        f["logenergy"][...] = updated
        f["logpress"][...] = updated_logpress

    return changed_slices, total_slices, changed_points, pressure_updated_points


def build_connector_curve(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Scan backward from T_max and extend lower T with a fixed matched slope."""
    point_slopes = np.gradient(y, x)

    transition_idx: int | None = None
    for idx in range(len(point_slopes) - 1, -1, -1):
        if point_slopes[idx] <= slope:
            if idx == len(point_slopes) - 1:
                transition_idx = idx
            else:
                hi = idx + 1
                if abs(point_slopes[hi] - slope) < abs(point_slopes[idx] - slope):
                    transition_idx = hi
                else:
                    transition_idx = idx
            break

    if transition_idx is None:
        return None

    connector = y.copy()
    x_match = float(x[transition_idx])
    y_match = float(y[transition_idx])
    left_mask = x < x_match
    connector[left_mask] = y_match + slope * (x[left_mask] - x_match)

    return connector, {
        "x_match": x_match,
        "y_match": y_match,
        "table_slope_at_match": float(point_slopes[transition_idx]),
        "connector_slope": slope,
        "transition_idx": transition_idx,
    }


def main() -> None:
    args = parse_args()

    if args.output_table is not None:
        if len(args.eos_files) != 1:
            raise ValueError("--output-table currently supports exactly one input EOS file")
        changed_slices, total_slices, changed_points, pressure_updated_points = (
            rewrite_logenergy_table(
            Path(args.eos_files[0]), args.output_table, args.connector_slope
            )
        )
        print(
            f"{args.output_table}: rewrote logenergy in {changed_slices} / "
            f"{total_slices} (Ye, rho) slices, updated {changed_points} logenergy "
            f"points, and updated {pressure_updated_points} logpress points"
        )

    show_delta_panel = args.draw_connector and not args.physical_axes
    if show_delta_panel:
        fig, (ax, delta_ax) = plt.subplots(
            2,
            1,
            figsize=(7.0, 6.0),
            sharex=True,
            gridspec_kw={"height_ratios": [4.0, 1.3]},
        )
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        delta_ax = None

    derivative_ax = delta_ax.twinx() if delta_ax is not None else None
    delta_handles = []
    derivative_handles = []

    for eos_file in args.eos_files:
        path = Path(eos_file)
        with h5py.File(path, "r") as f:
            logtemp = np.asarray(f["logtemp"][:], dtype=float)
            logenergy = np.asarray(f["logenergy"][:], dtype=float)
            ye = np.asarray(f["ye"][:], dtype=float)
            logrho = np.asarray(f["logrho"][:], dtype=float)
            energy_shift = float(np.asarray(f["energy_shift"][:]).reshape(-1)[0])

            if args.ye_index is None or args.rho_index is None:
                iy, ir = choose_slice(logenergy, args.slope_tol, args.min_rise)
            else:
                iy, ir = args.ye_index, args.rho_index

            if not (0 <= iy < len(ye)):
                raise IndexError(f"{path}: ye index {iy} is out of range [0, {len(ye) - 1}]")
            if not (0 <= ir < len(logrho)):
                raise IndexError(
                    f"{path}: rho index {ir} is out of range [0, {len(logrho) - 1}]"
                )

            curve = logenergy[iy, :, ir]
            label = (
                f"{label_from_path(path)} "
                f"(Ye[{iy}]={ye[iy]:.3f}, logrho[{ir}]={logrho[ir]:.3f})"
            )

            if args.physical_axes:
                temp = np.power(10.0, logtemp)
                eps = np.power(10.0, curve) - energy_shift
                positive = eps > 0.0
                ax.plot(
                    temp[positive],
                    eps[positive],
                    label=f"{label} table",
                    color="tab:blue",
                    linestyle="-",
                )
            else:
                ax.plot(
                    logtemp,
                    curve,
                    label=f"{label} table",
                    color="tab:blue",
                    linestyle="-",
                )

            if args.draw_connector:
                connector_result = build_connector_curve(
                    logtemp,
                    curve,
                    args.connector_slope,
                )
                if connector_result is not None:
                    connector_curve, connector_info = connector_result
                    modified_mask = logtemp <= connector_info["x_match"]
                    if args.physical_axes:
                        connector_eps = np.power(10.0, connector_curve) - energy_shift
                        positive = (connector_eps > 0.0) & modified_mask
                        ax.plot(
                            temp[positive],
                            connector_eps[positive],
                            label=f"{label} connector",
                            linestyle="--",
                            linewidth=2.0,
                            color="tab:orange",
                        )
                    else:
                        ax.plot(
                            logtemp[modified_mask],
                            connector_curve[modified_mask],
                            label=f"{label} connector",
                            linestyle="--",
                            linewidth=2.0,
                            color="tab:orange",
                        )
                        if delta_ax is not None:
                            derivative = np.gradient(curve, logtemp)
                            connector_derivative = np.gradient(connector_curve, logtemp)
                            (delta_line,) = delta_ax.plot(
                                logtemp[modified_mask],
                                connector_curve[modified_mask] - curve[modified_mask],
                                color="tab:red",
                                linestyle="--",
                                linewidth=2.0,
                                label=r"$\Delta$",
                            )
                            delta_handles.append(delta_line)
                            (table_slope_line,) = derivative_ax.plot(
                                logtemp,
                                derivative,
                                color="tab:blue",
                                linewidth=1.5,
                                alpha=0.9,
                                label="table slope",
                            )
                            (connector_slope_line,) = derivative_ax.plot(
                                logtemp[modified_mask],
                                connector_derivative[modified_mask],
                                color="tab:green",
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.9,
                                label="connector slope",
                            )
                            derivative_ax.set_ylabel("slope", color="tab:blue")
                            derivative_ax.tick_params(axis="y", colors="tab:blue")
                            delta_ax.axhline(0.0, color="0.5", linewidth=1.0, alpha=0.7)
                            derivative_ax.axhline(0.0, color="tab:blue", linewidth=1.0, alpha=0.25)
                            delta_ax.axvline(
                                connector_info["x_match"],
                                color="0.7",
                                linewidth=1.0,
                                alpha=0.8,
                            )
                            derivative_handles = [table_slope_line, connector_slope_line]
                    print(
                        f"{path}: connector slope={args.connector_slope:.3e}, "
                        f"match at log10(T)={connector_info['x_match']:.6g}, "
                        f"table slope at match={connector_info['table_slope_at_match']:.6g}"
                    )
                else:
                    print(f"{path}: connector skipped because no transition was detected")

            print(
                f"{path}: using Ye index {iy} (Ye={ye[iy]:.6g}), "
                f"rho index {ir} (log10(rho)={logrho[ir]:.6g})"
            )

    if args.physical_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T")
        ax.set_ylabel(r"$\epsilon$")
    else:
        ax.set_xlabel(r"$\log_{10}(T)$")
        ax.set_ylabel(r"$\log_{10}(\epsilon + \epsilon_0)$")
        if delta_ax is not None:
            delta_ax.set_xlabel(r"$\log_{10}(T)$")
            delta_ax.set_ylabel(r"$\Delta$", color="tab:red")
            delta_ax.tick_params(axis="y", colors="tab:red")
            delta_ax.grid(True, alpha=0.3)
            if delta_handles:
                delta_ax.legend(handles=delta_handles + derivative_handles, loc="upper left")

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
