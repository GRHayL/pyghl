from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyghl as ghl

from .common import set_flat_metric


def rel_err(expected: float, got: float) -> float:
    if expected != 0.0:
        return abs(1.0 - got / expected)
    return float(got != 0.0)


def compute_prim_errors(ref: ghl.Primitive, new: ghl.Primitive) -> float:
    err = 0.0
    err += rel_err(ref.rho, new.rho)
    err += rel_err(ref.temperature, new.temperature)
    err += rel_err(ref.press, new.press)
    err += rel_err(ref.vU[0], new.vU[0])
    err += rel_err(ref.vU[1], new.vU[1])
    err += rel_err(ref.vU[2], new.vU[2])
    return err / 6.0


def compute_x_from_prims(metric: ghl.Metric, prims: ghl.Primitive) -> float:
    h = 1.0 + prims.eps + prims.press / prims.rho
    W = prims.u0 * metric.lapse
    return h * W


def infer_default_output(model_path: Path) -> Path:
    try:
        import h5py

        with h5py.File(model_path, "r") as h5f:
            dims = h5f["dims"]
            n_hidden = int(dims["n_hidden"][()])
            hidden_dim = int(dims["hidden_dim"][()])
        return Path(f"nn_test_{n_hidden}x{hidden_dim}.asc")
    except Exception:
        return Path("nn_test.asc")


def infer_default_output_from_eos(eos_path: Path) -> Path:
    try:
        metadata = ghl.nn.eos_nn_metadata(eos_path)
        if metadata.get("contains_nn"):
            n_hidden = int(metadata["n_hidden"])
            hidden_dim = int(metadata["hidden_dim"])
            return Path(f"nn_test_{n_hidden}x{hidden_dim}.asc")
    except Exception:
        pass
    return Path("nn_test.asc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output

    params = ghl.initialize_params(
        main_routine=ghl.C2P_NONE,
        backup_routine=(ghl.C2P_NONE, ghl.C2P_NONE, ghl.C2P_NONE),
        evolve_entropy=False,
        evolve_temp=True,
        calc_prim_guess=True,
        psi6threshold=1e100,
        max_lorentz_factor=100.0,
        lorenz_damping_factor=0.0,
    )
    eos_init_kwargs = dict(
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
    loaded_from_eos = False
    try:
        eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
            params,
            args.table,
            enable_neural_net_c2p=True,
            **eos_init_kwargs,
        )
        loaded_from_eos = True
    except ghl.GRHayLError as exc:
        if args.model is None:
            raise SystemExit(
                f"{args.table} does not contain embedded neural-network data and no "
                f"--model fallback was provided.\nOriginal error: {exc}"
            ) from exc
        print(
            f"{args.table} does not contain readable embedded neural-network data; "
            f"falling back to {args.model}."
        )
        eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
            params,
            args.table,
            **eos_init_kwargs,
        )

    if args.model is not None:
        eos.load_nn_c2p_hdf5(str(args.model))

    if output is None:
        if loaded_from_eos:
            output = infer_default_output_from_eos(args.table)
        elif args.model is not None:
            output = infer_default_output(args.model)
        else:
            output = Path("nn_test.asc")
    metric, metric_aux = set_flat_metric()

    count = 0
    ghl_total_s = 0.0
    nn_total_s = 0.0
    with args.dataset.open("rb") as dataset_fp:
        _, _, n_blocks = ghl.nn.read_dataset_header(dataset_fp)
    if args.limit is not None:
        n_blocks = min(n_blocks, args.limit)
    report_progress_every = max(1, n_blocks // 100) if n_blocks else 1

    with output.open("w") as fp:
        fp.write("# Col. 1: GRHayL - Initial x Error vs. Exact\n")
        fp.write("# Col. 2: GRHayL - Initial Error vs. Orig\n")
        fp.write("# Col. 3: GRHayL - Final Error vs. Orig\n")
        fp.write("# Col. 4: GRHayL - Iterations\n")
        fp.write("# Col. 5: NN     - Initial x Error vs. Exact\n")
        fp.write("# Col. 6: NN     - Initial Error vs. Orig\n")
        fp.write("# Col. 7: NN     - Final Error vs. Orig\n")
        fp.write("# Col. 8: NN     - Iterations\n")

        for data in ghl.nn.iter_dataset_points(args.dataset):
            count += 1
            if args.limit is not None and count > args.limit:
                break
            if count % report_progress_every == 0:
                progress = min(100, count // report_progress_every)
                print(f"Progress: {progress:3d}%", end="\r", flush=True)

            rho = data.rho
            ye = data.ye
            temp = data.temp
            prs, eps, ent = eos.tabulated_compute_P_eps_S_from_T(rho, ye, temp)
            prims_orig = ghl.initialize_primitives(
                rho,
                prs,
                eps,
                data.vx,
                data.vy,
                data.vz,
                data.Bx,
                data.By,
                data.Bz,
                ent,
                ye,
                temp,
            )

            if temp > eos.T_max and abs(1.0 - temp / eos.T_max) > 1e-6:
                print(
                    "BEFORE Something weird just happened with temperature MAX: "
                    f"{temp:.15e}, {eos.T_max:.15e}"
                )

            if temp < eos.T_min and abs(1.0 - temp / eos.T_min) > 1e-6:
                print(
                    "BEFORE Something weird just happened with temperature MIN: "
                    f"{temp:.15e}, {eos.T_min:.15e}"
                )

            diagnostics = ghl.initialize_diagnostics()
            diagnostics.speed_limited = ghl.limit_v_and_compute_u0(params, metric, prims_orig)
            cons = ghl.compute_conservs(metric, metric_aux, prims_orig)
            cons_undens = ghl.undensitize_conservatives(metric.sqrt_detgamma, cons)

            prims_ghl = ghl.Primitive()
            prims_ghl.BU = (data.Bx, data.By, data.Bz)
            prims_nn = ghl.Primitive()
            prims_nn.temperature = eos.table_T_max
            prims_nn.BU = (data.Bx, data.By, data.Bz)

            prims_ghl = ghl.guess_primitives(params, eos, metric, cons_undens)
            x_ghl = compute_x_from_prims(metric, prims_ghl)
            prims_ghl.BU = (data.Bx, data.By, data.Bz)
            prims_ghl.temperature = prims_orig.temperature
            error_ghl = compute_prim_errors(prims_orig, prims_ghl)
            ghl_n_eos_inversions = None
            error_c2p_ghl = None
            start = time.perf_counter()
            try:
                eos.enable_neural_net_c2p = False
                ghl.tabulated_Palenzuela1D_energy(
                    params, eos, metric, metric_aux, cons_undens, prims_ghl, diagnostics
                )
                ghl_n_eos_inversions = diagnostics.n_iter
                error_c2p_ghl = compute_prim_errors(prims_orig, prims_ghl)
            except ghl.GRHayLError:
                pass
            ghl_total_s += time.perf_counter() - start

            x_guess = ghl.nn.nn_initial_guess(params, eos, metric, cons_undens, prims_nn)
            error_nn = compute_prim_errors(prims_orig, prims_nn)
            nn_n_eos_inversions = None
            error_c2p_nn = None
            start = time.perf_counter()
            try:
                eos.enable_neural_net_c2p = True
                ghl.tabulated_Palenzuela1D_energy(
                    params, eos, metric, metric_aux, cons_undens, prims_nn, diagnostics
                )
                nn_n_eos_inversions = diagnostics.n_iter
                error_c2p_nn = compute_prim_errors(prims_orig, prims_nn)
            except ghl.GRHayLError:
                pass
            nn_total_s += time.perf_counter() - start

            ghl_x_err = rel_err(data.x, x_ghl)
            nn_x_err = rel_err(data.x, x_guess)
            ghl_err_text = (
                f"{ghl_x_err:.8e} {error_ghl:.8e} {error_c2p_ghl:.8e} {ghl_n_eos_inversions:d}"
                if ghl_n_eos_inversions is not None
                else f"{ghl_x_err:.8e} NAN NAN NAN"
            )
            nn_err_text = (
                f"{nn_x_err:.8e} {error_nn:.8e} {error_c2p_nn:.8e} {nn_n_eos_inversions:d}"
                if nn_n_eos_inversions is not None
                else f"{nn_x_err:.8e} NAN NAN NAN"
            )
            fp.write(f"{ghl_err_text} {nn_err_text}\n")

    if count:
        print()
        print(f"Exec time ghl: total {ghl_total_s:g} s, avg {ghl_total_s / count * 1e6:g} us")
        print(f"Exec time nn : total {nn_total_s:g} s, avg {nn_total_s / count * 1e6:g} us")
        try:
            if loaded_from_eos:
                metadata = ghl.nn.eos_nn_metadata(args.table)
                n_hidden = int(metadata["n_hidden"])
                hidden_dim = int(metadata["hidden_dim"])
                out_dim = int(metadata.get("out_dim", 1))
            elif args.model is not None:
                import h5py

                with h5py.File(args.model, "r") as h5f:
                    dims = h5f["dims"]
                    n_hidden = int(dims["n_hidden"][()])
                    hidden_dim = int(dims["hidden_dim"][()])
                    out_dim = int(dims["out_dim"][()])
            else:
                raise ValueError("No neural-network metadata source available")
            print(
                f"Neural net size: 4"
                f"{' -> ' + ' -> '.join([str(hidden_dim)] * n_hidden)} -> {out_dim}"
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
