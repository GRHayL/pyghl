from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


NN_HDF5_FORMAT = "grhayl_nn_c2p_hdf5"
NN_HDF5_FORMAT_VERSION = 2
EOS_EMBED_GROUP = "grhayl_nn_c2p"
EOS_HASH_KIND = "md5-hdf5-content-v1"
DEFAULT_MODELS_SUBDIR = Path("nn_c2p") / "models"


def _update_hash_with_text(md5: hashlib._hashlib.HASH, text: str) -> None:
    md5.update(text.encode("utf-8"))
    md5.update(b"\0")


def _update_hash_with_bytes(md5: hashlib._hashlib.HASH, payload: bytes) -> None:
    md5.update(len(payload).to_bytes(8, "little", signed=False))
    md5.update(payload)


def _normalize_scalar(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float)):
        return repr(value).encode("ascii")
    return repr(value).encode("utf-8")


def _hash_attrs(md5: hashlib._hashlib.HASH, attrs: h5py.AttributeManager) -> None:
    for key in sorted(attrs.keys()):
        _update_hash_with_text(md5, f"attr:{key}")
        value = attrs[key]
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            _update_hash_with_text(md5, str(arr.dtype))
            _update_hash_with_text(md5, str(arr.shape))
            _update_hash_with_bytes(md5, arr.tobytes(order="C"))
        else:
            _update_hash_with_bytes(md5, _normalize_scalar(value))


def compute_hdf5_content_md5(
    path: str | Path,
    *,
    exclude_top_level: tuple[str, ...] = (EOS_EMBED_GROUP,),
) -> str:
    file_path = Path(path)
    md5 = hashlib.md5()
    with h5py.File(file_path, "r") as h5f:
        _hash_attrs(md5, h5f.attrs)
        dataset_names: list[str] = []

        def collect(name: str, obj: h5py.HLObject) -> None:
            top_level = name.split("/", 1)[0]
            if top_level in exclude_top_level:
                return
            if isinstance(obj, h5py.Dataset):
                dataset_names.append(name)

        h5f.visititems(collect)
        for name in sorted(dataset_names):
            ds = h5f[name]
            _update_hash_with_text(md5, f"dataset:{name}")
            _update_hash_with_text(md5, str(ds.dtype))
            _update_hash_with_text(md5, str(ds.shape))
            _hash_attrs(md5, ds.attrs)
            _update_hash_with_bytes(md5, np.asarray(ds[()]).tobytes(order="C"))
    return md5.hexdigest()


def compute_file_md5(path: str | Path) -> str:
    md5 = hashlib.md5()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def build_eos_metadata(eos_path: str | Path) -> dict[str, Any]:
    eos_file = Path(eos_path)
    return {
        "hash_kind": EOS_HASH_KIND,
        "canonical_md5": compute_hdf5_content_md5(eos_file),
        "file_md5": compute_file_md5(eos_file),
        "filename": eos_file.name,
        "path": str(eos_file.resolve()),
        "size_bytes": np.int64(eos_file.stat().st_size),
    }


def _utc_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_group_datasets(group: h5py.Group) -> dict[str, np.ndarray | np.generic]:
    return {name: group[name][()] for name in group.keys()}


def read_nn_hdf5_payload(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5f:
        fmt = h5f.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode("utf-8")
        if fmt != NN_HDF5_FORMAT:
            raise ValueError(f"{path!s} is not a {NN_HDF5_FORMAT!r} file.")

        payload: dict[str, Any] = {
            "dims": _read_group_datasets(h5f["dims"]),
            "meta": _read_group_datasets(h5f["meta"]),
            "scaling": _read_group_datasets(h5f["scaling"]),
            "layers": _read_group_datasets(h5f["layers"]),
            "audit": {},
        }

        audit_group = h5f.get("audit")
        if isinstance(audit_group, h5py.Group):
            if audit_group.keys():
                payload["audit"] = _read_group_datasets(audit_group)
            else:
                payload["audit"] = dict(audit_group.attrs.items())

        source_eos_group = h5f.get("source_eos")
        if isinstance(source_eos_group, h5py.Group):
            payload["source_eos"] = _read_group_datasets(source_eos_group)

    return payload


def installed_nn_models_dir() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_MODELS_SUBDIR


def iter_installed_nn_model_paths() -> list[Path]:
    models_dir = installed_nn_models_dir()
    if not models_dir.exists():
        return []
    return sorted(path for path in models_dir.glob("*.h5") if path.is_file())


def _read_model_source_eos_md5(model_path: str | Path) -> str | None:
    payload = read_nn_hdf5_payload(model_path)
    source_eos = payload.get("source_eos")
    if not source_eos:
        return None
    return _decode_hdf5_scalar(source_eos.get("canonical_md5"))


def find_matching_installed_nn_model(eos_path: str | Path) -> Path | None:
    eos_md5 = build_eos_metadata(eos_path)["canonical_md5"]
    for model_path in iter_installed_nn_model_paths():
        if _read_model_source_eos_md5(model_path) == eos_md5:
            return model_path
    return None


def installed_model_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_path in iter_installed_nn_model_paths():
        payload = read_nn_hdf5_payload(model_path)
        source_eos = payload.get("source_eos", {})
        summaries.append(
            {
                "model_path": str(model_path),
                "model_filename": model_path.name,
                "canonical_md5": _decode_hdf5_scalar(source_eos.get("canonical_md5", "")),
                "source_eos_filename": _decode_hdf5_scalar(source_eos.get("filename", "")),
                "hidden_dim": int(payload["dims"]["hidden_dim"]),
                "n_hidden": int(payload["dims"]["n_hidden"]),
            }
        )
    return summaries


def eos_nn_metadata(eos_path: str | Path) -> dict[str, Any]:
    eos_file = Path(eos_path)
    info: dict[str, Any] = {
        "eos_path": str(eos_file.resolve()),
        "eos_filename": eos_file.name,
        "group_name": EOS_EMBED_GROUP,
        "contains_nn": False,
    }
    with h5py.File(eos_file, "r") as h5f:
        if EOS_EMBED_GROUP not in h5f:
            return info
        root = h5f[EOS_EMBED_GROUP]
        info["contains_nn"] = True
        info["format"] = _decode_hdf5_scalar(root["format"][()])
        info["format_version"] = int(root["format_version"][()])
        info["nn_hdf5_filename"] = _decode_hdf5_scalar(root["nn_hdf5_filename"][()])
        info["nn_hdf5_md5"] = _decode_hdf5_scalar(root["nn_hdf5_md5"][()])
        info["eos_hash_kind"] = _decode_hdf5_scalar(root["eos_hash_kind"][()])

        dims = root.get("dims")
        if isinstance(dims, h5py.Group):
            info["in_dim"] = int(dims["in_dim"][()])
            info["hidden_dim"] = int(dims["hidden_dim"][()])
            info["n_hidden"] = int(dims["n_hidden"][()])
            info["out_dim"] = int(dims["out_dim"][()])

        provenance = root.get("provenance")
        if isinstance(provenance, h5py.Group):
            for key in provenance.keys():
                value = provenance[key][()]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (bytes, np.bytes_, np.ndarray)):
                    value = _decode_hdf5_scalar(value)
                info[key] = value
    return info


def _write_scalar_dataset(group: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        dt = h5py.string_dtype("utf-8")
        group.create_dataset(name, data=value, dtype=dt)
        return
    if isinstance(value, bytes):
        dt = h5py.string_dtype("utf-8")
        group.create_dataset(name, data=value.decode("utf-8"), dtype=dt)
        return
    group.create_dataset(name, data=value)


def _write_mapping(group: h5py.Group, mapping: dict[str, Any]) -> None:
    for key, value in sorted(mapping.items()):
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, np.generic):
            group.create_dataset(key, data=value)
        else:
            _write_scalar_dataset(group, key, value)


def write_nn_hdf5(
    payload: dict[str, Any],
    output_path: str | Path,
    *,
    root_attrs: dict[str, Any] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        attrs = {
            "format": NN_HDF5_FORMAT,
            "format_version": np.int32(NN_HDF5_FORMAT_VERSION),
        }
        if root_attrs:
            attrs.update(root_attrs)
        for key, value in attrs.items():
            h5f.attrs[key] = value

        for section in ("dims", "meta", "scaling", "layers"):
            _write_mapping(h5f.create_group(section), payload[section])

        audit = payload.get("audit", {})
        if audit:
            audit_group = h5f.create_group("audit")
            for key, value in sorted(audit.items()):
                if isinstance(value, str):
                    audit_group.attrs[key] = value
                elif isinstance(value, bytes):
                    audit_group.attrs[key] = value.decode("utf-8")
                elif isinstance(value, np.generic):
                    audit_group.attrs[key] = value.item()
                else:
                    audit_group.attrs[key] = value

        source_eos = payload.get("source_eos")
        if source_eos:
            _write_mapping(h5f.create_group("source_eos"), source_eos)


def append_nn_to_eos_file(
    eos_path: str | Path,
    nn_hdf5_path: str | Path,
    *,
    overwrite: bool = False,
    require_eos_match: bool = True,
) -> dict[str, Any]:
    eos_file = Path(eos_path)
    nn_file = Path(nn_hdf5_path)
    payload = read_nn_hdf5_payload(nn_file)
    eos_metadata = build_eos_metadata(eos_file)
    source_eos = payload.get("source_eos")
    existing_info = eos_nn_metadata(eos_file)

    if require_eos_match:
        if not source_eos:
            raise ValueError(
                f"{nn_file!s} does not contain source EOS metadata; rerun training with an EOS file "
                "or use --force to append without verification."
            )
        expected_kind = _decode_hdf5_scalar(source_eos["hash_kind"])
        expected_md5 = _decode_hdf5_scalar(source_eos["canonical_md5"])
        if expected_kind != EOS_HASH_KIND:
            raise ValueError(
                f"Unsupported EOS hash kind {expected_kind!r} in {nn_file!s}; expected {EOS_HASH_KIND!r}."
            )
        if expected_md5 != eos_metadata["canonical_md5"]:
            raise ValueError(
                f"EOS mismatch: model expects canonical md5 {expected_md5}, "
                f"but {eos_file!s} hashes to {eos_metadata['canonical_md5']}."
            )

    if existing_info["contains_nn"] and not overwrite:
        raise ValueError(
            f"{eos_file!s} already contains {EOS_EMBED_GROUP!r}; "
            "rerun with overwrite=True to replace it (CLI: --overwrite)."
        )

    with h5py.File(eos_file, "a") as h5f:
        if EOS_EMBED_GROUP in h5f:
            del h5f[EOS_EMBED_GROUP]
        root = h5f.create_group(EOS_EMBED_GROUP)
        _write_scalar_dataset(root, "format", NN_HDF5_FORMAT)
        _write_scalar_dataset(root, "format_version", np.int32(NN_HDF5_FORMAT_VERSION))
        _write_scalar_dataset(root, "nn_hdf5_md5", compute_file_md5(nn_file))
        _write_scalar_dataset(root, "nn_hdf5_filename", nn_file.name)
        _write_scalar_dataset(root, "eos_hash_kind", EOS_HASH_KIND)

        for section in ("dims", "meta", "scaling", "layers"):
            _write_mapping(root.create_group(section), payload[section])

        audit = payload.get("audit", {})
        if audit:
            _write_mapping(root.create_group("audit"), audit)

        provenance = dict(eos_metadata)
        provenance["embedded_utc"] = _utc_now_timestamp()
        if source_eos:
            provenance["source_model_eos_md5"] = _decode_hdf5_scalar(source_eos["canonical_md5"])
            provenance["source_model_eos_filename"] = _decode_hdf5_scalar(source_eos["filename"])
        _write_mapping(root.create_group("provenance"), provenance)
    summary = eos_nn_metadata(eos_file)
    summary["overwrite_performed"] = bool(existing_info["contains_nn"])
    return summary


def append_matching_installed_nn_to_eos_file(
    eos_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_path = find_matching_installed_nn_model(eos_path)
    if model_path is None:
        eos_md5 = build_eos_metadata(eos_path)["canonical_md5"]
        available = installed_model_summaries()
        available_md5s = sorted({item["canonical_md5"] for item in available if item["canonical_md5"]})
        raise ValueError(
            f"No installed neural-network model matches this EOS; canonical md5 is {eos_md5}. "
            f"Known installed EOS hashes: {available_md5s}"
        )
    return append_nn_to_eos_file(eos_path, model_path, overwrite=overwrite, require_eos_match=True)


def remove_nn_from_eos_file(eos_path: str | Path) -> dict[str, Any]:
    eos_file = Path(eos_path)
    before = build_eos_metadata(eos_file)
    existing = eos_nn_metadata(eos_file)
    if not existing["contains_nn"]:
        raise ValueError(
            f"{eos_file!s} does not contain embedded neural-network data group {EOS_EMBED_GROUP!r}."
        )

    with h5py.File(eos_file, "a") as h5f:
        del h5f[EOS_EMBED_GROUP]

    after = build_eos_metadata(eos_file)
    return {
        "eos_filename": eos_file.name,
        "eos_path": str(eos_file.resolve()),
        "group_name": EOS_EMBED_GROUP,
        "removed_nn_hdf5_filename": existing.get("nn_hdf5_filename"),
        "raw_md5_before": before["file_md5"],
        "raw_md5_after": after["file_md5"],
        "canonical_md5_before": before["canonical_md5"],
        "canonical_md5_after": after["canonical_md5"],
    }


def _sanitize_eos_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("EOS name must contain at least one alphanumeric character.")
    return cleaned


def install_nn_model(
    nn_hdf5_path: str | Path,
    *,
    eos_name: str,
    overwrite: bool = False,
) -> Path:
    src = Path(nn_hdf5_path)
    if not src.is_file():
        raise FileNotFoundError(f"NN model file not found: {src!s}")
    sanitized = _sanitize_eos_name(eos_name)
    models_dir = installed_nn_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / f"{sanitized}.h5"
    if dest.exists() and not overwrite:
        raise ValueError(
            f"Installed model {dest.name} already exists; rerun with overwrite=True to replace it."
        )
    shutil.copy2(src, dest)
    return dest


def _decode_hdf5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_hdf5_scalar(value[()])
        raise TypeError(f"Expected scalar HDF5 value, got shape {value.shape}.")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)
