from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Iterator

if TYPE_CHECKING:
    from . import _pyghl


@dataclass(frozen=True)
class DatasetPoint:
    rho: float
    temp: float
    ye: float
    W: float
    log_PmagoP: float
    vx: float
    vy: float
    vz: float
    Bx: float
    By: float
    Bz: float
    q: float
    r: float
    s: float
    t: float
    x: float


@dataclass(frozen=True)
class NNGuessInput:
    q: float
    r: float
    s: float
    t: float


@dataclass(frozen=True)
class NNGuess:
    x: float


def guess(eos: _pyghl.TabulatedEOS, q: float, r: float, s: float, t: float) -> NNGuess:
    _pyghl = _require_pyghl()
    x = _pyghl.nn_c2p_guess(eos, float(q), float(r), float(s), float(t))
    return NNGuess(x=float(x))


def guess_x(eos: _pyghl.TabulatedEOS, q: float, r: float, s: float, t: float) -> float:
    return guess(eos, q, r, s, t).x


def flat_metric() -> tuple[_pyghl.Metric, _pyghl.ADMAux]:
    _pyghl = _require_pyghl()
    metric = _pyghl.initialize_metric(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    return metric, _pyghl.compute_ADM_auxiliaries(metric)


def nn_initial_guess(
    params: _pyghl.Params,
    eos: _pyghl.TabulatedEOS,
    metric: _pyghl.Metric,
    cons_undens: _pyghl.Conservative,
    prims: _pyghl.Primitive,
) -> float:
    _pyghl = _require_pyghl()
    SU, B_squared, S_squared, BdotS = _pyghl.compute_SU_Bsq_Ssq_BdotS(
        metric, cons_undens, prims
    )

    invD = 1.0 / cons_undens.rho
    q = cons_undens.tau * invD
    r = S_squared * invD * invD
    s = B_squared * invD
    t = BdotS / math.pow(cons_undens.rho, 1.5)

    nn_guess = guess(eos, q, r, s, t)
    x = nn_guess.x
    x_lo = 1.0 + q - s
    x_hi = 2.0 + 2.0 * q - s
    x = min(max(x, x_lo), x_hi)

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
    _pyghl.limit_utilde_and_compute_v(params, metric, utildeU, prims)
    prims.press, prims.eps, prims.entropy = eos.tabulated_compute_P_eps_S_from_T(
        prims.rho, prims.Y_e, prims.temperature
    )
    return x


def _require_pyghl():
    from . import require_bindings

    require_bindings()
    from . import _pyghl

    return _pyghl


def read_dataset_header(fp: BinaryIO) -> tuple[int, int, int]:
    payload = fp.read(24)
    if len(payload) != 24:
        raise EOFError("dataset header is incomplete")
    return struct.unpack("<QQQ", payload)


def iter_dataset_points(path: str | Path) -> Iterator[DatasetPoint]:
    with Path(path).open("rb") as fp:
        sizeof_float, n_floats_per_block, n_blocks = read_dataset_header(fp)
        if sizeof_float != 4:
            raise ValueError(f"expected 4-byte floats, got {sizeof_float}")
        if n_floats_per_block != 16:
            raise ValueError(f"expected 16 floats per block, got {n_floats_per_block}")

        block_struct = struct.Struct("<16f")
        for _ in range(n_blocks):
            payload = fp.read(block_struct.size)
            if len(payload) != block_struct.size:
                raise EOFError("dataset ended mid-block")
            yield DatasetPoint(*block_struct.unpack(payload))


def _load_training_api():
    try:
        from . import _nn_dataset, _nn_infer, _nn_train
    except ImportError as exc:
        raise ImportError(
            "Neural-network training helpers require optional dependencies. "
            "Install with `python -m pip install -e ./python[nn]`."
        ) from exc
    return _nn_dataset, _nn_infer, _nn_train


def read_training_dataset(path: str | Path, **kwargs):
    dataset_mod, _, _ = _load_training_api()
    return dataset_mod.read_training_dataset(path, **kwargs)


def train_regressor(*args, **kwargs):
    _, _, train_mod = _load_training_api()
    return train_mod.train_regressor(*args, **kwargs)


def train_on_dataset(dataset_path: str | Path, **kwargs):
    _, _, train_mod = _load_training_api()
    return train_mod.train_on_dataset(dataset_path, **kwargs)


def export_to_c_header(*args, **kwargs):
    _, _, train_mod = _load_training_api()
    return train_mod.export_to_c_header(*args, **kwargs)


def export_to_hdf5(*args, **kwargs):
    _, _, train_mod = _load_training_api()
    return train_mod.export_to_hdf5(*args, **kwargs)


def append_to_eos_file(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.append_nn_to_eos_file(*args, **kwargs)


def append_matching_installed_to_eos_file(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.append_matching_installed_nn_to_eos_file(*args, **kwargs)


def remove_from_eos_file(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.remove_nn_from_eos_file(*args, **kwargs)


def eos_nn_metadata(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.eos_nn_metadata(*args, **kwargs)


def installed_nn_models(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.installed_model_summaries(*args, **kwargs)


def find_matching_installed_model(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.find_matching_installed_nn_model(*args, **kwargs)


def install_nn_model(*args, **kwargs):
    from . import _nn_hdf5

    return _nn_hdf5.install_nn_model(*args, **kwargs)


def save_inference_bundle(*args, **kwargs):
    _, infer_mod, _ = _load_training_api()
    return infer_mod.save_inference_bundle(*args, **kwargs)


def load_inference_bundle(*args, **kwargs):
    _, infer_mod, _ = _load_training_api()
    return infer_mod.load_inference_bundle(*args, **kwargs)
