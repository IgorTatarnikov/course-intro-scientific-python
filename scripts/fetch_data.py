"""Download the course datasets into ``data/``.

Learners do not need to run this: the files are committed to the repository so
nothing has to be downloaded during class. It exists so instructors can see
where every file came from and rebuild ``data/`` from scratch.

Run from the repository root, inside the course environment. The IBL slice
streams from the DANDI archive and also needs ``remfile``:

    pip install remfile
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
    """The moon-landing image (SPL, CC BY 4.0), used on the Day 2 slides."""
    (DATA / "moonlanding.png").write_bytes(download(SPL_RAW + "moonlanding.png"))


def fetch_gapminder() -> None:
    """Gapminder CSVs (Software Carpentry, CC BY 4.0), used on the slides."""
    archive = zipfile.ZipFile(io.BytesIO(download(GAPMINDER_ZIP)))
    for member in archive.namelist():
        if Path(member).name in ("gapminder_gdp_europe.csv", "gapminder_all.csv"):
            (DATA / Path(member).name).write_bytes(archive.read(member))


IBL_DANDISET = "000409"
IBL_VERSION = "0.260309.1324"
IBL_RAW = "99911f99-85d0-4f2d-bf3e-0cd15bee5bf7"  # sub-NYU-46 ..._desc-raw_ecephys.nwb (51 GB)
IBL_PROCESSED = "196aa923-79c8-4524-a1b2-344fc30d8cb2"  # ..._desc-processed_behavior+ecephys.nwb
IBL_LFP_START = 618.0  # seconds into the session; the window holds trials 88 to 90
IBL_LFP_SECONDS = 10.0
IBL_LFP_RATE = 500  # Hz after decimating from 2500 Hz; keeps the 60 Hz mains hum

# Allen atlas acronyms for the brain areas this probe passes through.
IBL_AREAS = {
    "Piriform area": "PIR",
    "Agranular insular area posterior part": "AIp",
    "Gustatory areas": "GU",
    "Visceral area": "VISC",
    "Supplemental somatosensory area": "SSs",
    "Primary somatosensory area nose": "SSp-n",
    "void": "void",
}


def fetch_ibl_session() -> None:
    """A small slice of one IBL Brain Wide Map session (DANDI 000409, CC BY 4.0).

    One Neuropixels probe in a mouse doing the IBL visual decision task: 10 s of
    LFP from all 384 sites, the atlas location of every site, the 533 trials,
    the spike-sorted units and spike counts around each stimulus.

    Streams from DANDI, so it needs ``pip install remfile`` on top of the
    course environment. Reading the spike times pulls about 120 MB, and the
    whole function takes a few minutes.
    """
    import h5py
    import numpy as np
    import remfile
    import scipy.signal

    def open_asset(asset_id):
        url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
        return h5py.File(remfile.File(url), "r")

    def text(dataset):
        return dataset[:].astype(str)

    raw = open_asset(IBL_RAW)
    processed = open_asset(IBL_PROCESSED)

    # LFP: find the window with the timestamps, decimate 2500 -> 500 Hz, and
    # store int16 with the same scaling as the original (4.6875 uV per unit).
    series = raw["acquisition/ElectricalSeriesProbe01LF"]
    print("Reading the LFP window")
    timestamps = series["timestamps"][:3_000_000]
    first = int(np.searchsorted(timestamps, IBL_LFP_START))
    n_raw = int(IBL_LFP_SECONDS * 2500)
    pad = 2500  # 1 s each side, so the filter has no edge effects
    lfp = series["data"][first - pad : first + n_raw + pad].astype(np.float64)
    lfp = scipy.signal.decimate(lfp, 2500 // IBL_LFP_RATE, axis=0, ftype="fir")
    edge = pad * IBL_LFP_RATE // 2500
    lfp = np.round(lfp[edge:-edge]).astype(np.int16)
    lfp_start = timestamps[first]

    electrodes = raw["general/extracellular_ephys/electrodes"]
    location = text(electrodes["location"])
    area = np.array([IBL_AREAS[loc.split(" layer")[0]] for loc in location])
    layer = np.array([loc.split(" layer ")[1] if " layer " in loc else "" for loc in location])

    print("Reading trials and units")
    trials = processed["intervals/trials"]
    stim_on = trials["gabor_stimulus_onset_time"][:]
    side = np.where(text(trials["gabor_stimulus_side"]) == "right", 1, -1)
    # Turning the wheel counter-clockwise moves the stimulus right.
    choice = np.where(text(trials["mouse_wheel_choice"]) == "counter_clockwise", 1, -1)

    units = processed["units"]
    unit_label = text(units["kilosort2_label"])
    unit_channel = units["max_electrode"][:]

    # Spike counts of the "good" units in 50 ms bins from 0.5 s before to 1 s
    # after each stimulus.
    print("Reading spike times")
    spike_times = units["spike_times"][:]
    ends = units["spike_times_index"][:]
    starts = np.concatenate([[0], ends[:-1]])
    good = np.flatnonzero(unit_label == "good")
    bin_edges = np.arange(-0.5, 1.0 + 1e-9, 0.05)
    spike_counts = np.zeros((good.size, stim_on.size, bin_edges.size - 1), dtype=np.uint8)
    for row, unit in enumerate(good):
        times = spike_times[starts[unit] : ends[unit]]
        for trial, onset in enumerate(stim_on):
            spike_counts[row, trial] = np.histogram(times - onset, bins=bin_edges)[0]

    out = DATA / "ibl_session.h5"
    with h5py.File(out, "w") as f:
        f.attrs["description"] = (
            "One Neuropixels probe in a mouse doing the IBL visual decision task. "
            "A small slice of IBL Brain Wide Map session 64e3fb86 (subject NYU-46)."
        )
        f.attrs["source"] = (
            f"DANDI:{IBL_DANDISET}/{IBL_VERSION}, assets {IBL_RAW} and {IBL_PROCESSED}"
        )
        f.attrs["license"] = "CC BY 4.0"
        f.attrs["citation"] = (
            "International Brain Laboratory et al. (2026) IBL - Brain Wide Map "
            f"(Version {IBL_VERSION}) [Data set]. DANDI Archive. "
            f"https://doi.org/10.48324/dandi.{IBL_DANDISET}/{IBL_VERSION}"
        )
        compress = {"compression": "gzip", "compression_opts": 9, "shuffle": True}

        def put(group, name, data, **kwargs):
            # track_times=False keeps the bytes identical between runs, so
            # regenerating the file does not show up as a change in git.
            return group.create_dataset(name, data=data, track_times=False, **kwargs)

        g = f.create_group("lfp", track_order=True)
        put(g, "data", lfp, **compress)
        g["data"].attrs["dims"] = "time, channel"
        g["data"].attrs["unit"] = "raw ADC counts; multiply by uV_per_count for microvolts"
        g.attrs["uV_per_count"] = 4.6875
        g.attrs["sampling_rate_Hz"] = IBL_LFP_RATE
        g.attrs["start_time_s"] = lfp_start

        g = f.create_group("electrodes", track_order=True)
        put(g, "channel", np.arange(location.size))
        put(g, "area", area.astype("S"))
        put(g, "layer", layer.astype("S"))
        put(g, "region", location.astype("S"))
        put(g, "x_um", electrodes["rel_x"][:])
        put(g, "depth_um", electrodes["rel_y"][:])
        g["depth_um"].attrs["description"] = "distance above the probe tip"

        # track_order keeps the columns in this order rather than alphabetical.
        g = f.create_group("trials", track_order=True)
        put(g, "stim_on_s", stim_on)
        # Adding 0.0 turns -0.0 (zero contrast "on the left") into 0.0.
        put(g, "contrast", side * trials["gabor_stimulus_contrast"][:] + 0.0)
        g["contrast"].attrs["description"] = "percent; negative = left, positive = right"
        put(g, "probability_left", trials["probability_left"][:])
        put(g, "choice", choice)
        g["choice"].attrs["description"] = "-1 = turned the stimulus left, 1 = right"
        put(g, "correct", trials["is_mouse_rewarded"][:])
        put(g, "response_time_s", trials["choice_registration_time"][:] - stim_on)

        g = f.create_group("units", track_order=True)
        put(g, "unit", units["id"][:])
        put(g, "channel", unit_channel)
        put(g, "area", area[unit_channel].astype("S"))
        put(g, "depth_um", electrodes["rel_y"][:][unit_channel])
        put(g, "label", unit_label.astype("S"))
        put(g, "firing_rate_Hz", units["firing_rate"][:])
        put(g, "amplitude_uV", units["median_spike_amplitude_uV"][:])
        put(g, "spike_width_ms", units["peak_to_trough_duration_ms"][:])
        put(g, "presence_ratio", units["presence_ratio"][:])

        g = f.create_group("spike_counts", track_order=True)
        put(g, "data", spike_counts, **compress)
        g["data"].attrs["dims"] = "unit, trial, bin"
        put(g, "unit", units["id"][:][good])
        put(g, "bin_start_s", bin_edges[:-1])
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch_spl()
    fetch_gapminder()
    fetch_ibl_session()
    print("Done:", sorted(p.name for p in DATA.iterdir()))
