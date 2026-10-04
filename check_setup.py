"""Check that the course environment is ready.

Run this from the course folder, with the environment activated:

    conda activate scientific-python
    python check_setup.py

Every line should say OK. If anything says MISSING or FAILED, copy the whole
output and send it to the instructors before the course starts.
"""

import importlib
import os
import shutil
import sys
from pathlib import Path

PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("h5py", "h5py"),
    ("jupyterlab", "jupyterlab"),
]

# Command-line tools used in the environments block. conda is a shell function
# once initialised, so fall back to CONDA_EXE, which conda sets on activation.
TOOLS = {
    "conda": lambda: shutil.which("conda") or os.environ.get("CONDA_EXE"),
    "uv": lambda: shutil.which("uv"),
}

DATA_FILES = [
    "ibl_session.h5",
    "moonlanding.png",
    "gapminder_gdp_europe.csv",
    "gapminder_all.csv",
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


def check_tools() -> bool:
    ok = True
    for name, find in TOOLS.items():
        if path := find():
            print(f"  OK       {name:<14} {path}")
        else:
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


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    if sys.version_info < (3, 11):
        print("  FAILED   Python 3.11 or newer is needed. Is the course environment active?")
    print("Packages:")
    results = [check_packages()]
    print("\nTools:")
    results.append(check_tools())
    print("\nData files:")
    results.append(check_data())

    if all(results):
        print("\nAll good: you are ready for the course.")
    else:
        print("\nSomething is missing. Copy this output and send it to the instructors.")
        sys.exit(1)
