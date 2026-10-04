"""Check that the course environment is ready.

Run this from the course folder, with the environment activated:

    conda activate scientific-python
    python check_setup.py

Every line should say OK. If anything says MISSING or FAILED, copy the whole
output and send it to the instructors before the course starts.
"""

import importlib
import sys
from pathlib import Path

PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-image", "skimage"),
    ("xarray", "xarray"),
    ("netCDF4", "netCDF4"),
    ("dask", "dask"),
    ("napari", "napari"),
    ("jupyterlab", "jupyterlab"),
]

DATA_FILES = [
    "populations.txt",
    "moonlanding.png",
    "face.png",
    "human_mitosis.png",
    "gapminder_gdp_europe.csv",
    "gapminder_all.csv",
    "air_temperature_daily.nc",
]


def check_packages() -> bool:
    ok = True
    for name, module in PACKAGES:
        try:
            version = importlib.import_module(module).__version__
            print(f"  OK       {name:<14} {version}")
        except ImportError:
            print(f"  MISSING  {name}")
            ok = False
    return ok


def check_data() -> bool:
    data = Path(__file__).resolve().parent / "data"
    ok = True
    for name in DATA_FILES:
        if (data / name).exists():
            print(f"  OK       data/{name}")
        else:
            print(f"  MISSING  data/{name}")
            ok = False
    return ok


def check_napari() -> bool:
    """napari needs a working Qt backend to open a window."""
    try:
        from qtpy import QtCore  # noqa: F401

        print("  OK       Qt backend for napari")
        return True
    except Exception as err:  # Qt failures raise a variety of error types
        print(f"  FAILED   Qt backend for napari: {err}")
        return False


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    if sys.version_info < (3, 11):
        print("  FAILED   Python 3.11 or newer is needed. Is the course environment active?")
    print("Packages:")
    results = [check_packages()]
    print("\nData files:")
    results.append(check_data())
    print("\nnapari:")
    results.append(check_napari())

    if all(results):
        print("\nAll good: you are ready for the course.")
    else:
        print("\nSomething is missing. Copy this output and send it to the instructors.")
        sys.exit(1)
