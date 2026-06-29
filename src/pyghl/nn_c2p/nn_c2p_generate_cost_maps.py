from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

import pyghl as ghl

from .._nn_hdf5 import build_eos_metadata
from .common import set_flat_metric
from .nn_c2p_generate_dataset import compute_cons_error, reconstruct_prims_from_x


ROLE_UNIFORM = 1 << 0
ROLE_X_LO = 1 << 1
ROLE_X_HI = 1 << 2
ROLE_MIDPOINT = 1 << 3
ROLE_PHASE_A = 1 << 4


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_params() -> ghl.Params:
    return ghl.initialize_params(
        main_routine=ghl.C2P_NOBLE2D,
        backup_routine=(ghl.C2P_NONE, ghl.C2P_NONE, ghl.C2P_NONE),
        evolve_entropy=False,
        evolve_temp=True,
        calc_prim_guess=False,
        psi6threshold=1e100,
        max_lorentz_factor=100.0,
        lorenz_damping_factor=0.0,
    )


def make_eos(
    params: ghl.Params,
    table: str | Path,
    *,
    enable_nn: bool = False,
) -> ghl.TabulatedEOS:
    return ghl.eos.initialize_tabulated_eos_functions_and_params(
        params,
        table,
        rho_atm=1e-12,
        rho_min=1e-12,
        rho_max=5e-3,
        Ye_atm=0.5,
        Ye_min=0.05,
        Ye_max=0.5,
        T_atm=1e-2,
        T_min=1e-2,
        T_max=90.0,
        root_finding_precision=1e-10,
        enable_neural_net_c2p=enable_nn,
    )


def sample_state(
    rng: random.Random,
    eos: ghl.TabulatedEOS,
) -> dict[str, float]:
    lr_min = math.log10(eos.rho_min)
    lr_max = math.log10(eos.rho_max)
    lt_min = math.log10(eos.T_min)
    lt_max = math.log10(eos.T_max)

    rho = 10.0 ** rng.uniform(lr_min, lr_max)
    temp = 10.0 ** rng.uniform(lt_min, lt_max)
    ye = rng.uniform(eos.Ye_min, eos.Ye_max)
    W = rng.uniform(1.0, 10.0)
    log_PmagoP = rng.uniform(-12.0, -3.0)

    prs, eps, ent = eos.tabulated_compute_P_eps_S_from_T(rho, ye, temp)
    vmag = math.sqrt(max(0.0, 1.0 - 1.0 / (W * W)))
    Bmag = math.sqrt(2.0 * (10.0**log_PmagoP) * prs)

    phi = 2.0 * math.pi * rng.uniform(0.0, 1.0)
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    if abs(vmag) > 1.0e-15:
        vx = vmag * sin_theta * math.cos(phi)
        vy = vmag * sin_theta * math.sin(phi)
        vz = vmag * cos_theta
        Bx = -Bmag * vx / vmag
        By = -Bmag * vy / vmag
        Bz = -Bmag * vz / vmag
    else:
        vx = vy = vz = 0.0
        Bx = Bmag * sin_theta * math.cos(phi)
        By = Bmag * sin_theta * math.sin(phi)
        Bz = Bmag * cos_theta

    return {
        "rho": rho,
        "temp": temp,
        "ye": ye,
        "W": W,
        "log_PmagoP": log_PmagoP,
        "prs": prs,
        "eps": eps,
        "ent": ent,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
    }


def evaluate_candidate(
    params: ghl.Params,
    eos: ghl.TabulatedEOS,
    metric: ghl.Metric,
    metric_aux: ghl.ADMAux,
    cons_undens: ghl.Conservative,
    cons_ref: ghl.Conservative,
    B_fields: tuple[float, float, float],
    x: float,
) -> dict[str, float | int | bool]:
    out: dict[str, float | int | bool] = {
        "seed_ok": False,
        "success": False,
        "n_iter": -1,
        "failure_code": 1,
        "cons_error": float("nan"),
        "residual_norm": float("nan"),
        "physical_flags": 0,
    }
    try:
        prims_guess = reconstruct_prims_from_x(
            params, eos, metric, cons_undens, B_fields, x
        )
        out["seed_ok"] = True
    except Exception:
        return out

    diagnostics = ghl.initialize_diagnostics()
    try:
        ghl.tabulated_con2prim_multi_method(
            params, eos, metric, metric_aux, cons_undens, prims_guess, diagnostics
        )
        cons_recovered = ghl.compute_conservs(metric, metric_aux, prims_guess)
        cons_recovered_undens = ghl.undensitize_conservatives(
            metric.sqrt_detgamma, cons_recovered
        )
        out["success"] = True
        out["failure_code"] = 0
        out["n_iter"] = int(diagnostics.n_iter)
        out["cons_error"] = float(compute_cons_error(cons_ref, cons_recovered_undens))
    except ghl.GRHayLError:
        out["n_iter"] = int(getattr(diagnostics, "n_iter", -1))
    return out


def compute_phase_a_guess(
    eos: ghl.TabulatedEOS,
    q: float,
    r: float,
    s: float,
    t: float,
) -> float:
    return float(ghl.nn.guess_x(eos, q, r, s, t))


def load_phase_a_model(
    eos: ghl.TabulatedEOS,
    phase_a_model: str | Path | None,
) -> bool:
    if phase_a_model is None:
        return bool(getattr(eos, "enable_neural_net_c2p", False))
    eos.load_nn_c2p_hdf5(str(phase_a_model))
    return True


def generate_cost_maps(
    table: str | Path,
    *,
    output: str | Path,
    n_states: int = 10000,
    scan_points: int = 65,
    seed: int = 0,
    include_phase_a: bool = True,
    phase_a_model: str | Path | None = None,
    sampling: str = "random",
) -> Path:
    if int(n_states) < 1:
        raise ValueError("n_states must be >= 1.")
    if int(scan_points) < 2:
        raise ValueError("scan_points must be >= 2.")
    if sampling != "random":
        raise ValueError(
            f"Unsupported sampling mode {sampling!r}. "
            "Initial implementation supports 'random' only."
        )

    params = make_params()
    eos = make_eos(params, table)
    metric, metric_aux = set_flat_metric()
    rng = random.Random(int(seed))

    phase_a_loaded = False
    if include_phase_a:
        if phase_a_model is not None:
            eos.load_nn_c2p_hdf5(str(phase_a_model))
            phase_a_loaded = True
        else:
            try:
                eos_embedded = make_eos(params, table, enable_nn=True)
                eos = eos_embedded
                phase_a_loaded = True
            except ghl.GRHayLError:
                phase_a_loaded = False
                print("Warning: could not load embedded Phase A model; disabling phase_a baseline.")

    N = int(n_states)
    K = int(scan_points)
    y_uniform = np.linspace(0.0, 1.0, K, dtype=np.float32)

    q_arr = np.empty(N, dtype=np.float32)
    r_arr = np.empty(N, dtype=np.float32)
    s_arr = np.empty(N, dtype=np.float32)
    t_arr = np.empty(N, dtype=np.float32)
    x_lo_arr = np.empty(N, dtype=np.float32)
    x_hi_arr = np.empty(N, dtype=np.float32)
    x_exact_arr = np.empty(N, dtype=np.float32)
    y_exact_arr = np.empty(N, dtype=np.float32)
    rho_arr = np.empty(N, dtype=np.float32)
    temp_arr = np.empty(N, dtype=np.float32)
    ye_arr = np.empty(N, dtype=np.float32)
    W_arr = np.empty(N, dtype=np.float32)
    logpmagop_arr = np.empty(N, dtype=np.float32)
    vx_arr = np.empty(N, dtype=np.float32)
    vy_arr = np.empty(N, dtype=np.float32)
    vz_arr = np.empty(N, dtype=np.float32)
    Bx_arr = np.empty(N, dtype=np.float32)
    By_arr = np.empty(N, dtype=np.float32)
    Bz_arr = np.empty(N, dtype=np.float32)

    uniform_x = np.empty((N, K), dtype=np.float32)
    uniform_role_mask = np.full((N, K), ROLE_UNIFORM, dtype=np.uint32)
    uniform_seed_ok = np.zeros((N, K), dtype=np.bool_)
    uniform_success = np.zeros((N, K), dtype=np.bool_)
    uniform_n_iter = np.full((N, K), -1, dtype=np.int32)
    uniform_failure_code = np.ones((N, K), dtype=np.int32)
    uniform_cons_error = np.full((N, K), np.nan, dtype=np.float32)
    uniform_residual = np.full((N, K), np.nan, dtype=np.float32)
    uniform_physical_flags = np.zeros((N, K), dtype=np.uint32)

    baseline_names = np.array(
        ["x_lo", "midpoint", "x_hi", "phase_a", "x_exact"], dtype="S16"
    )
    n_baselines = len(baseline_names)
    baseline_y = np.full((N, n_baselines), np.nan, dtype=np.float32)
    baseline_x = np.full((N, n_baselines), np.nan, dtype=np.float32)
    baseline_seed_ok = np.zeros((N, n_baselines), dtype=np.bool_)
    baseline_success = np.zeros((N, n_baselines), dtype=np.bool_)
    baseline_n_iter = np.full((N, n_baselines), -1, dtype=np.int32)
    baseline_failure_code = np.ones((N, n_baselines), dtype=np.int32)
    baseline_cons_error = np.full((N, n_baselines), np.nan, dtype=np.float32)
    baseline_residual = np.full((N, n_baselines), np.nan, dtype=np.float32)
    baseline_physical_flags = np.zeros((N, n_baselines), dtype=np.uint32)

    report_every = max(1, N // 100)
    for idx in range(N):
        if (idx + 1) % report_every == 0 or idx == N - 1:
            progress = int(100.0 * (idx + 1) / N)
            print(f"Progress: {progress:3d}%", end="\r", flush=True)

        state = sample_state(rng, eos)
        prims = ghl.initialize_primitives(
            state["rho"],
            state["prs"],
            state["eps"],
            state["vx"],
            state["vy"],
            state["vz"],
            state["Bx"],
            state["By"],
            state["Bz"],
            state["ent"],
            state["ye"],
            state["temp"],
        )
        ghl.limit_v_and_compute_u0(params, metric, prims)
        cons = ghl.compute_conservs(metric, metric_aux, prims)
        cons_undens = ghl.undensitize_conservatives(metric.sqrt_detgamma, cons)
        _, B_squared, S_squared, BdotS = ghl.compute_SU_Bsq_Ssq_BdotS(
            metric, cons_undens, prims
        )

        invD = 1.0 / cons_undens.rho
        q = cons_undens.tau * invD
        r = S_squared * invD * invD
        s = B_squared * invD
        t = BdotS / (cons_undens.rho**1.5)
        x_lo = 1.0 + q - s
        x_hi = 2.0 + 2.0 * q - s
        width = x_hi - x_lo
        h = 1.0 + prims.eps + prims.press / prims.rho
        x_exact = h * prims.u0
        y_exact = 0.0 if width <= 0.0 else (x_exact - x_lo) / width

        q_arr[idx] = q
        r_arr[idx] = r
        s_arr[idx] = s
        t_arr[idx] = t
        x_lo_arr[idx] = x_lo
        x_hi_arr[idx] = x_hi
        x_exact_arr[idx] = x_exact
        y_exact_arr[idx] = y_exact
        rho_arr[idx] = state["rho"]
        temp_arr[idx] = state["temp"]
        ye_arr[idx] = state["ye"]
        W_arr[idx] = state["W"]
        logpmagop_arr[idx] = state["log_PmagoP"]
        vx_arr[idx] = state["vx"]
        vy_arr[idx] = state["vy"]
        vz_arr[idx] = state["vz"]
        Bx_arr[idx] = state["Bx"]
        By_arr[idx] = state["By"]
        Bz_arr[idx] = state["Bz"]

        B_fields = (state["Bx"], state["By"], state["Bz"])
        for j, y in enumerate(y_uniform):
            x = x_lo + float(y) * width
            uniform_x[idx, j] = x
            if j == 0:
                uniform_role_mask[idx, j] |= ROLE_X_LO
            if j == K - 1:
                uniform_role_mask[idx, j] |= ROLE_X_HI
            if j == K // 2 and abs(float(y) - 0.5) <= (0.5 / max(1, K - 1)):
                uniform_role_mask[idx, j] |= ROLE_MIDPOINT

            result = evaluate_candidate(
                params,
                eos,
                metric,
                metric_aux,
                cons_undens,
                cons_undens,
                B_fields,
                x,
            )
            uniform_seed_ok[idx, j] = bool(result["seed_ok"])
            uniform_success[idx, j] = bool(result["success"])
            uniform_n_iter[idx, j] = int(result["n_iter"])
            uniform_failure_code[idx, j] = int(result["failure_code"])
            uniform_cons_error[idx, j] = np.float32(result["cons_error"])
            uniform_residual[idx, j] = np.float32(result["residual_norm"])
            uniform_physical_flags[idx, j] = np.uint32(result["physical_flags"])

        baseline_specs = [
            (0, 0.0, x_lo),
            (1, 0.5, x_lo + 0.5 * width),
            (2, 1.0, x_hi),
        ]
        if phase_a_loaded:
            try:
                x_phase_a_raw = compute_phase_a_guess(eos, q, r, s, t)
                y_phase_a = max(0.0, min(1.0, (x_phase_a_raw - x_lo) / width))
                x_phase_a = x_lo + y_phase_a * width
                baseline_specs.append((3, y_phase_a, x_phase_a))
            except Exception:
                pass
        baseline_specs.append((4, y_exact, x_exact))

        for baseline_id, yb, xb in baseline_specs:
            baseline_y[idx, baseline_id] = yb
            baseline_x[idx, baseline_id] = xb
            result = evaluate_candidate(
                params,
                eos,
                metric,
                metric_aux,
                cons_undens,
                cons_undens,
                B_fields,
                xb,
            )
            baseline_seed_ok[idx, baseline_id] = bool(result["seed_ok"])
            baseline_success[idx, baseline_id] = bool(result["success"])
            baseline_n_iter[idx, baseline_id] = int(result["n_iter"])
            baseline_failure_code[idx, baseline_id] = int(result["failure_code"])
            baseline_cons_error[idx, baseline_id] = np.float32(result["cons_error"])
            baseline_residual[idx, baseline_id] = np.float32(result["residual_norm"])
            baseline_physical_flags[idx, baseline_id] = np.uint32(result["physical_flags"])

    print()
    output = Path(output)
    with h5py.File(output, "w") as h5f:
        meta = h5f.create_group("meta")
        meta.attrs["created_utc"] = utc_now()
        meta.attrs["table"] = str(Path(table))
        meta.attrs["seed"] = int(seed)
        meta.attrs["n_states"] = N
        meta.attrs["scan_points"] = K
        meta.attrs["sampling"] = sampling
        meta.attrs["include_phase_a"] = bool(phase_a_loaded)
        if phase_a_model is not None:
            meta.attrs["phase_a_model"] = str(Path(phase_a_model))
        eos_meta = build_eos_metadata(table)
        meta.attrs["eos_filename"] = eos_meta["filename"]
        meta.attrs["eos_hash"] = eos_meta["canonical_md5"]
        meta.attrs["rho_min"] = 1.0e-12
        meta.attrs["rho_max"] = 5.0e-3
        meta.attrs["Ye_min"] = 0.05
        meta.attrs["Ye_max"] = 0.5
        meta.attrs["T_min"] = 1.0e-2
        meta.attrs["T_max"] = 90.0
        meta.attrs["W_max"] = 10.0

        states = h5f.create_group("states")
        for name, arr in (
            ("q", q_arr),
            ("r", r_arr),
            ("s", s_arr),
            ("t", t_arr),
            ("x_lo", x_lo_arr),
            ("x_hi", x_hi_arr),
            ("x_exact", x_exact_arr),
            ("y_exact", y_exact_arr),
            ("rho", rho_arr),
            ("T", temp_arr),
            ("Ye", ye_arr),
            ("W", W_arr),
            ("log_PmagoP", logpmagop_arr),
            ("vx", vx_arr),
            ("vy", vy_arr),
            ("vz", vz_arr),
            ("Bx", Bx_arr),
            ("By", By_arr),
            ("Bz", Bz_arr),
        ):
            states.create_dataset(name, data=arr)

        grid = h5f.create_group("grid")
        grid.create_dataset("y_uniform", data=y_uniform)

        uniform = h5f.create_group("uniform")
        uniform.create_dataset("y", data=np.broadcast_to(y_uniform[None, :], (N, K)))
        uniform.create_dataset("x", data=uniform_x)
        uniform.create_dataset("role_mask", data=uniform_role_mask)
        uniform.create_dataset("seed_ok", data=uniform_seed_ok)
        uniform.create_dataset("success", data=uniform_success)
        uniform.create_dataset("n_iter", data=uniform_n_iter)
        uniform.create_dataset("failure_code", data=uniform_failure_code)
        uniform.create_dataset("cons_error", data=uniform_cons_error)
        uniform.create_dataset("residual_norm", data=uniform_residual)
        uniform.create_dataset("physical_flags", data=uniform_physical_flags)

        baselines = h5f.create_group("baselines")
        baselines.create_dataset("names", data=baseline_names)
        baselines.create_dataset("y", data=baseline_y)
        baselines.create_dataset("x", data=baseline_x)
        baselines.create_dataset("seed_ok", data=baseline_seed_ok)
        baselines.create_dataset("success", data=baseline_success)
        baselines.create_dataset("n_iter", data=baseline_n_iter)
        baselines.create_dataset("failure_code", data=baseline_failure_code)
        baselines.create_dataset("cons_error", data=baseline_cons_error)
        baselines.create_dataset("residual_norm", data=baseline_residual)
        baselines.create_dataset("physical_flags", data=baseline_physical_flags)

        # Suggested compatibility alias for the implementation plan.
        candidates = h5f.create_group("candidates")
        candidates.create_dataset("y", data=np.broadcast_to(y_uniform[None, :], (N, K)))
        candidates.create_dataset("x", data=uniform_x)
        candidates.create_dataset("role_mask", data=uniform_role_mask)
        candidates.create_dataset("seed_ok", data=uniform_seed_ok)
        candidates.create_dataset("success", data=uniform_success)
        candidates.create_dataset("n_iter", data=uniform_n_iter)
        candidates.create_dataset("failure_code", data=uniform_failure_code)
        candidates.create_dataset("cons_error", data=uniform_cons_error)
        candidates.create_dataset("residual_norm", data=uniform_residual)
        candidates.create_dataset("physical_flags", data=uniform_physical_flags)

    print(f"Wrote solver cost maps to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_states", type=int, default=10000)
    parser.add_argument("--scan_points", type=int, default=65)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("random",), default="random")
    parser.add_argument("--include_phase_a", choices=("yes", "no"), default="yes")
    parser.add_argument("--phase_a_model", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_cost_maps(
        args.table,
        output=args.output,
        n_states=args.n_states,
        scan_points=args.scan_points,
        seed=args.seed,
        include_phase_a=args.include_phase_a == "yes",
        phase_a_model=args.phase_a_model,
        sampling=args.sampling,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
