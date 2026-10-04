"""Download the course datasets into ``data/``.

Learners do not need to run this: the files are committed to the repository so
nothing has to be downloaded during class. It exists so instructors can see
where every file came from and rebuild ``data/`` from scratch.

Run from the repository root, inside the course environment:

    python scripts/fetch_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

SPL_RAW = (
    "https://raw.githubusercontent.com/scipy-lectures/"
    "scientific-python-lectures/main/data/"
)
GAPMINDER_ZIP = (
    "https://swcarpentry.github.io/python-novice-gapminder/files/"
    "python-novice-gapminder-data.zip"
)


def download(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()


def fetch_spl() -> None:
    """Hare/lynx populations and the moon-landing image (SPL, CC BY 4.0)."""
    for name in ["populations.txt", "moonlanding.png"]:
        (DATA / name).write_bytes(download(SPL_RAW + name))


def fetch_gapminder() -> None:
    """Gapminder GDP CSVs (Software Carpentry, CC BY 4.0)."""
    archive = zipfile.ZipFile(io.BytesIO(download(GAPMINDER_ZIP)))
    for member in archive.namelist():
        if member.endswith(".csv"):
            (DATA / Path(member).name).write_bytes(archive.read(member))


def fetch_images() -> None:
    """Raccoon face (SciPy) and dividing human cells (scikit-image, CC0)."""
    import scipy.datasets
    import skimage.data
    import skimage.io

    skimage.io.imsave(DATA / "face.png", scipy.datasets.face())
    skimage.io.imsave(DATA / "human_mitosis.png", skimage.data.human_mitosis())


def fetch_air_temperature() -> None:
    """NCEP air temperature over North America, as used by the xarray tutorial.

    The full tutorial file covers 2013-2014 at 6-hourly resolution. We keep
    daily means so the file stays small.
    """
    import xarray as xr

    ds = xr.tutorial.open_dataset("air_temperature")
    daily = ds.resample(time="1D").mean().astype("float32")
    daily.attrs = ds.attrs
    daily["air"].attrs = ds["air"].attrs | {
        "long_name": "Daily mean air temperature at sigma level 995"
    }
    daily.attrs["title"] = "Daily means of the 4x daily NMC reanalysis (2013-2014)"
    daily.to_netcdf(
        DATA / "air_temperature_daily.nc",
        # Stored in 100-day chunks, which the dask notebook's chunks={"time": 100} matches.
        encoding={"air": {"zlib": True, "complevel": 4, "chunksizes": (100, 25, 53)}},
    )


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch_spl()
    fetch_gapminder()
    fetch_images()
    fetch_air_temperature()
    print("Done:", sorted(p.name for p in DATA.iterdir()))
