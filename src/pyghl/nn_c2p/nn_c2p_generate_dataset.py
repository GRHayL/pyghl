from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

import pyghl as ghl

from .common import set_flat_metric


def rel_err(expected: float, got: float) -> float:
    if expected != 0.0:
        return abs(1.0 - got / expected)
    return float(got != 0.0)


def compute_cons_error(ref: ghl.Conservative, got: ghl.Conservative) -> float:
    err = 0.0
    err += rel_err(ref.rho, got.rho)
    err += rel_err(ref.tau, got.tau)
    for i in range(3):
        err += rel_err(ref.SD[i], got.SD[i])
    return err / 5.0


def reconstruct_prims_from_x(
    params: ghl.Params,
    eos: ghl.TabulatedEOS,
    metric: ghl.Metric,
    cons_undens: ghl.Conservative,
    B_fields: tuple[float, float, float],
    x_in: float,
) -> ghl.Primitive:
    prims = ghl.Primitive()
    prims.BU = B_fields

    SU, B_squared, S_squared, BdotS = ghl.compute_SU_Bsq_Ssq_BdotS(
        metric, cons_undens, prims
    )
    invD = 1.0 / cons_undens.rho
    q = cons_undens.tau * invD
    r = S_squared * invD * invD
    s = B_squared * invD
    t = BdotS / (cons_undens.rho**1.5)
    x_lo = 1.0 + q - s
    x_hi = 2.0 + 2.0 * q - s
    x = min(max(float(x_in), x_lo), x_hi)

    Wminus2 = 1.0 - (x * x * r + (2.0 * x + s) * t * t) / (x * x * (x + s) * (x + s))
    Wminus2 = min(max(Wminus2, params.max_lorentz_factor**-2), 1.0)
    W = Wminus2**-0.5

    prims.rho = cons_undens.rho / W
    prims.Y_e = cons_undens.Y_e / cons_undens.rho
    prims.eps = (
        W
        - 1.0
        + (1.0 - W * W) * x / W
        + W * (q - s + t * t / (2.0 * x * x) + s / (2.0 * W * W))
    )
    prims.rho, prims.Y_e, prims.eps = eos.tabulated_enforce_bounds_rho_Ye_eps(
        prims.rho, prims.Y_e, prims.eps
    )
    prims.press, prims.temperature = eos.tabulated_compute_P_T_from_eps(
        prims.rho, prims.Y_e, prims.eps
    )

    Z = x * prims.rho * W
    utildeU = tuple(
        W * (SU[i] + BdotS * prims.BU[i] / Z) / (Z + B_squared) for i in range(3)
    )
    prims.rho, prims.Y_e, prims.temperature = eos.tabulated_enforce_bounds_rho_Ye_T(
        prims.rho, prims.Y_e, prims.temperature
    )
    ghl.limit_utilde_and_compute_v(params, metric, utildeU, prims)
    prims.press, prims.eps, prims.entropy = eos.tabulated_compute_P_eps_S_from_T(
        prims.rho, prims.Y_e, prims.temperature
    )
    return prims


def rank_candidate(
    success: bool,
    n_iter: int,
    cons_err: float,
) -> tuple[int, int, float]:
    return (0 if success else 1, int(n_iter), float(cons_err))


def scan_x_best(
    params: ghl.Params,
    eos: ghl.TabulatedEOS,
    metric: ghl.Metric,
    metric_aux: ghl.ADMAux,
    cons_undens: ghl.Conservative,
    cons_ref: ghl.Conservative,
    B_fields: tuple[float, float, float],
    q: float,
    s: float,
    *,
    scan_points: int,
) -> float:
    x_lo = 1.0 + q - s
    x_hi = 2.0 + 2.0 * q - s
    width = x_hi - x_lo
    if not math.isfinite(width) or width <= 0.0:
        return x_lo

    best_key: tuple[int, int, float] | None = None
    best_x = x_lo
    for j in range(scan_points):
        y = 0.0 if scan_points == 1 else j / (scan_points - 1)
        x = x_lo + y * width
        try:
            prims_guess = reconstruct_prims_from_x(
                params, eos, metric, cons_undens, B_fields, x
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
            cons_err = compute_cons_error(cons_ref, cons_recovered_undens)
            key = rank_candidate(True, diagnostics.n_iter, cons_err)
        except ghl.GRHayLError:
            key = rank_candidate(False, int(1e9), float("inf"))

        if best_key is None or key < best_key:
            best_key = key
            best_x = x
    return best_x


def generate_dataset(
    table: str | Path,
    dataset_type: str,
    *,
    n_pts: int = 16,
    output: str | Path | None = None,
    target_mode: str = "x_correction",
    scan_points: int = 17,
    c2p_method: int = ghl.C2P_NOBLE2D,
) -> Path:
    table = Path(table)
    is_test_dataset = dataset_type == "test"
    if dataset_type not in ("train", "test"):
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    rng = random.Random(42 if is_test_dataset else 0)

    if target_mode not in ("x_correction", "x_best_correction"):
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}. "
            "Expected 'x_correction' or 'x_best_correction'."
        )
    n_pts = int(n_pts)
    if n_pts < 2:
        raise ValueError("n_pts must be >= 2.")
    if int(scan_points) < 1:
        raise ValueError("scan_points must be >= 1.")

    params = ghl.initialize_params(
        main_routine=c2p_method if target_mode == "x_best_correction" else ghl.C2P_NONE,
        backup_routine=(ghl.C2P_NONE, ghl.C2P_NONE, ghl.C2P_NONE),
        evolve_entropy=False,
        evolve_temp=True,
        calc_prim_guess=(target_mode != "x_best_correction"),
        psi6threshold=1e100,
        max_lorentz_factor=100.0,
        lorenz_damping_factor=0.0,
    )
    eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
        params,
        table,
        rho_atm=1e-12,
        rho_min=1e-12,
        rho_max=1e-3,
        Ye_atm=0.5,
        Ye_min=0.05,
        Ye_max=0.5,
        T_atm=1e-2,
        T_min=1e-2,
        T_max=1e2,
        root_finding_precision=1e-10,
    )
    metric, metric_aux = set_flat_metric()

    n_blocks = n_pts**5
    output = Path(output) if output is not None else Path(
        "nn_test_dataset.bin" if is_test_dataset else "nn_training_dataset.bin"
    )
    block_struct = struct.Struct("<16f")

    lr_min = math.log10(eos.rho_min)
    lr_max = math.log10(eos.rho_max)
    lt_min = math.log10(eos.T_min)
    lt_max = math.log10(eos.T_max)
    ye_min = eos.Ye_min
    ye_max = eos.Ye_max
    W_min = 1.0
    W_max = 10.0
    log_PmagoP_min = -12.0
    log_PmagoP_max = -3.0

    dlr = (lr_max - lr_min) / (n_pts - 1)
    dlt = (lt_max - lt_min) / (n_pts - 1)
    dye = (ye_max - ye_min) / (n_pts - 1)
    dW = (W_max - W_min) / (n_pts - 1)
    dPmag = (log_PmagoP_max - log_PmagoP_min) / (n_pts - 1)

    mins = {key: float("inf") for key in ("q", "r", "s", "t", "x")}
    maxs = {key: float("-inf") for key in ("q", "r", "s", "t", "x")}

    report_progress_every = max(1, n_blocks // 100)
    with output.open("wb") as fp:
        fp.write(struct.pack("<QQQ", 4, 16, n_blocks))
        count = 0
        for n_rho in range(n_pts):
            for n_t in range(n_pts):
                for n_ye in range(n_pts):
                    for n_W in range(n_pts):
                        for n_Pmag in range(n_pts):
                            count += 1
                            if count % report_progress_every == 0:
                                progress = min(100, count // report_progress_every)
                                print(f"Progress: {progress:3d}%", end="\r", flush=True)
                            if is_test_dataset:
                                rho = 10.0 ** rng.uniform(lr_min, lr_max)
                                temp = 10.0 ** rng.uniform(lt_min, lt_max)
                                ye = rng.uniform(ye_min, ye_max)
                                W = rng.uniform(W_min, W_max)
                                log_PmagoP = rng.uniform(log_PmagoP_min, log_PmagoP_max)
                            else:
                                rho = 10.0 ** (lr_min + n_rho * dlr)
                                temp = 10.0 ** (lt_min + n_t * dlt)
                                ye = ye_min + n_ye * dye
                                W = W_min + n_W * dW
                                log_PmagoP = log_PmagoP_min + n_Pmag * dPmag

                            if temp > eos.T_max and abs(1.0 - temp / eos.T_max) > 1e-8:
                                print(
                                    "Something weird just happened with temperature MAX: "
                                    f"{temp:.15e}, {eos.T_max:.15e}"
                                )

                            if temp < eos.T_min and abs(1.0 - temp / eos.T_min) > 1e-8:
                                print(
                                    "Something weird just happened with temperature MIN: "
                                    f"{temp:.15e}, {eos.T_min:.15e}"
                                )

                            prs, eps, ent = eos.tabulated_compute_P_eps_S_from_T(
                                rho, ye, temp
                            )
                            v = math.sqrt(1.0 - 1.0 / (W * W))
                            B = math.sqrt(2.0 * 10.0**log_PmagoP * prs)

                            phi = 2.0 * math.pi * rng.uniform(0.0, 1.0)
                            cos_theta = rng.uniform(-1.0, 1.0)
                            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

                            if abs(v) > 1e-15:
                                vx = v * sin_theta * math.cos(phi)
                                vy = v * sin_theta * math.sin(phi)
                                vz = v * cos_theta
                                Bx = -B * vx / v
                                By = -B * vy / v
                                Bz = -B * vz / v
                            else:
                                vx = vy = vz = 0.0
                                Bx = B * sin_theta * math.cos(phi)
                                By = B * sin_theta * math.sin(phi)
                                Bz = B * cos_theta

                            prims = ghl.initialize_primitives(
                                rho, prs, eps, vx, vy, vz, Bx, By, Bz, ent, ye, temp
                            )
                            diagnostics = ghl.initialize_diagnostics()
                            diagnostics.speed_limited = ghl.limit_v_and_compute_u0(
                                params, metric, prims
                            )

                            cons = ghl.compute_conservs(metric, metric_aux, prims)
                            cons_undens = ghl.undensitize_conservatives(
                                metric.sqrt_detgamma, cons
                            )
                            _, B_squared, S_squared, BdotS = ghl.compute_SU_Bsq_Ssq_BdotS(
                                metric, cons_undens, prims
                            )

                            invD = 1.0 / cons_undens.rho
                            h = 1.0 + prims.eps + prims.press / prims.rho
                            q = cons_undens.tau * invD
                            r = S_squared * invD * invD
                            s = B_squared * invD
                            t = BdotS / (cons_undens.rho**1.5)
                            x_exact = h * W
                            x_target = x_exact
                            if target_mode == "x_best_correction":
                                x_target = scan_x_best(
                                    params,
                                    eos,
                                    metric,
                                    metric_aux,
                                    cons_undens,
                                    cons_undens,
                                    (Bx, By, Bz),
                                    q,
                                    s,
                                    scan_points=scan_points,
                                )

                            for key, value in {"q": q, "r": r, "s": s, "t": t, "x": x_target}.items():
                                mins[key] = min(mins[key], value)
                                maxs[key] = max(maxs[key], value)

                            total = (
                                rho
                                + temp
                                + ye
                                + W
                                + log_PmagoP
                                + vx
                                + vy
                                + vz
                                + Bx
                                + By
                                + Bz
                                + q
                                + r
                                + s
                                + t
                                + x_target
                            )
                            if not math.isfinite(total):
                                raise ValueError("encountered a non-finite dataset value")

                            fp.write(
                                block_struct.pack(
                                    rho,
                                    temp,
                                    ye,
                                    W,
                                    log_PmagoP,
                                    vx,
                                    vy,
                                    vz,
                                    Bx,
                                    By,
                                    Bz,
                                    q,
                                    r,
                                    s,
                                    t,
                                    x_target,
                                )
                            )

    print()
    print("The following brackets were found for this dataset:")
    for key in ("q", "r", "s", "t", "x"):
        print(f"{key} in [{mins[key]:+.7e}, {maxs[key]:+.7e}]")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("dataset_type", type=str, choices=("train", "test"))
    parser.add_argument("--n_pts", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--target_mode",
        choices=("x_correction", "x_best_correction"),
        default="x_correction",
    )
    parser.add_argument("--scan_points", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_dataset(
        args.table,
        args.dataset_type,
        n_pts=args.n_pts,
        output=args.output,
        target_mode=args.target_mode,
        scan_points=args.scan_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
