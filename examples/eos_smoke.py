import pyghl as ghl


def main() -> None:
    params = ghl.initialize_params()

    eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
        params,
        "~/codes/eos_tables/SLy4_3335_rho391_temp163_ye66.h5",
    )

    rho = 1e-12
    ye = 0.05
    temp = 1e2

    rho, ye, temp = eos.tabulated_enforce_bounds_rho_Ye_T(rho, ye, temp)
    pressure = eos.tabulated_compute_P_from_T(rho, ye, temp)

    print(f"rho={rho:.6e}, Ye={ye:.6e}, T={temp:.6e}, P={pressure:.6e}")


if __name__ == "__main__":
    main()
