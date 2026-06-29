from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PYTHON_DIR = Path(__file__).resolve().parent


def _default_grhayl_root() -> Path:
    env_root = os.environ.get("GRHAYL_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()

    submodule_root = PYTHON_DIR / "extern" / "GRHayL"
    if submodule_root.exists():
        return submodule_root

    return submodule_root


GRHAYL_ROOT = _default_grhayl_root()
BUILD_LIB_DIR = GRHAYL_ROOT / "build" / "lib"


def _run_make_grhayl() -> None:
    makefile = GRHAYL_ROOT / "Makefile"
    build_dir = GRHAYL_ROOT / "build"
    if not makefile.exists() or not build_dir.exists():
        configure = GRHAYL_ROOT / "configure"
        if not configure.exists():
            raise RuntimeError(
                "Could not find GRHayL configure script. Initialize extern/GRHayL "
                "or set GRHAYL_DIR to a configured GRHayL checkout."
            )
        configure_args = ["--prefix=."]
        if makefile.exists():
            configure_args.append("--reconfigure")
        configure_args.extend(shlex.split(os.environ.get("GRHAYL_CONFIGURE_ARGS", "")))
        subprocess.check_call([str(configure), *configure_args], cwd=GRHAYL_ROOT)
    subprocess.check_call(["make", "-C", str(GRHAYL_ROOT), "grhayl"])


class BuildExt(build_ext):
    def run(self) -> None:
        _run_make_grhayl()
        super().run()

        # For wheel/non-editable builds, ship libghl next to the extension so
        # the extension can resolve it through $ORIGIN rpath.
        if not self.inplace:
            ext_path = Path(self.get_ext_fullpath("pyghl._pyghl")).resolve()
            target_dir = ext_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            for libname in ("libghl.so", "libghl_1.0.0.so"):
                src = BUILD_LIB_DIR / libname
                if src.exists():
                    shutil.copy2(src, target_dir / libname)


RPATH_ARGS = [
    f"-Wl,-rpath,{BUILD_LIB_DIR}",
    "-Wl,-rpath,$ORIGIN",
]

ext_modules = [
    Extension(
        "pyghl._pyghl",
        sources=["csrc/pyghl_module.c"],
        include_dirs=[
            str(GRHAYL_ROOT / "GRHayL" / "include"),
            str(GRHAYL_ROOT / "Unit_Tests" / "data_gen"),
        ],
        library_dirs=[str(BUILD_LIB_DIR)],
        libraries=["ghl"],
        extra_compile_args=["-std=c99"],
        extra_link_args=RPATH_ARGS,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
