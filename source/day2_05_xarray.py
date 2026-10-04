# %% [markdown]
# # Day 2 · Whirlwind tour: xarray
#
# **Block:** whirlwind tour, xarray (about 10 min: demo, then try it)
#
# **Problem it solves:** real N-D data has *named* dimensions (time, channel;
# or unit, trial, time) with *coordinates* along them. NumPy only knows axis 0,
# 1, 2 and positions. xarray keeps the names and coordinates attached, so you
# can write `lfp.sel(time=slice(622.5, 624))` instead of working out
# `lfp[2250:3000]`, and `lfp.groupby("area")` instead of building masks.
#
# The demo turns our recording from Day 1 into labelled arrays. The slides used
# the air temperature data from the [xarray tutorial](https://tutorial.xarray.dev);
# the ideas are the same.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

with h5py.File("../data/ibl_session.h5") as f:
    raw = f["lfp/data"][:]
    uV_per_count = f["lfp"].attrs["uV_per_count"]
    sampling_rate = f["lfp"].attrs["sampling_rate_Hz"]
    lfp_start = f["lfp"].attrs["start_time_s"]
    area = f["electrodes/area"][:].astype(str)
    depth_um = f["electrodes/depth_um"][:]
    spike_counts = f["spike_counts/data"][:]
    spike_unit = f["spike_counts/unit"][:]
    bin_start = f["spike_counts/bin_start_s"][:]
    unit_area = f["units/area"][:].astype(str)
    stim_on = f["trials/stim_on_s"][:]
    contrast = f["trials/contrast"][:]
    correct = f["trials/correct"][:]

# %% [markdown]
# ## Demo
#
# A `DataArray` is a NumPy array plus **dimension names**, **coordinates**
# (labels along each dimension) and **attributes**. A coordinate can also
# label a dimension without being its index: here every channel has an `area`
# and a `depth`. In Jupyter, click the icons in the output to see them.

# %%
lfp = xr.DataArray(
    raw * uV_per_count,
    dims=("time", "channel"),
    coords={
        "time": lfp_start + np.arange(raw.shape[0]) / sampling_rate,  # seconds into the session
        "channel": np.arange(raw.shape[1]),
        "area": ("channel", area),
        "depth": ("channel", depth_um),
    },
    name="LFP",
    attrs={"units": "µV"},
)
lfp

# %% [markdown]
# **Select by label**, not by position. Times are seconds into the session, so
# the stimulus times from the trials table can be used directly.
# `method="nearest"` snaps to the closest sample:

# %%
lfp.sel(channel=200).sel(time=slice(stim_on[89] - 0.5, stim_on[89] + 1)).plot(figsize=(8, 3))
plt.axvline(stim_on[89], color="grey", linestyle="--");

# %%
lfp.sel(time=stim_on[89], method="nearest").plot(y="depth", figsize=(3, 5));

# %% [markdown]
# A boolean coordinate selects too: every channel in primary somatosensory
# cortex, without counting which columns those are:

# %%
lfp.sel(channel=lfp.area == "SSp-n")

# %% [markdown]
# **Reduce over a named dimension.** No more remembering that time is axis 0:

# %%
lfp.std(dim="time").plot(y="depth", figsize=(3, 5));

# %% [markdown]
# **Group by** a coordinate. Averaging the two sites at each depth gives one
# row per depth, which shows the whole probe as an image:

# %%
lfp.groupby("depth").mean().plot(x="time", y="depth", cmap="RdBu_r", robust=True, figsize=(10, 4));

# %% [markdown]
# Spike counts have three named dimensions. Coordinates can describe each
# trial (its contrast, and whether it was correct) and each unit (its area):

# %%
counts = xr.DataArray(
    spike_counts,
    dims=("unit", "trial", "time"),
    coords={
        "unit": spike_unit,
        "trial": np.arange(spike_counts.shape[1]),
        "time": bin_start + 0.025,  # centre of each bin, seconds from the stimulus
        "area": ("unit", unit_area[spike_unit]),
        "strength": ("trial", np.abs(contrast)),
        "correct": ("trial", correct),
    },
    name="spike count",
)
counts

# %% [markdown]
# The population response for each stimulus strength, in one line: average
# over units, then group the trials by strength and average within each group.

# %%
rate = counts.mean("unit").groupby("strength").mean() / 0.05
rate.name, rate.attrs["units"] = "firing rate", "spikes/s"
rate.plot.line(x="time", hue="strength", figsize=(7, 3));

# %% [markdown]
# The rise after the stimulus is about the same whatever its contrast, a hint
# that these neurons respond to the mouse's movement rather than to what it sees.

# %% [markdown]
# ## Try it
#
# 1. Plot the mean LFP over the channels in `"SSs"` and over the channels in
#    `"SSp-n"`, from 0.5 s before to 1 s after trial 90's stimulus, on one plot.
#    (Hint: `groupby("area").mean()`, then `.sel` for both the time and the
#    areas.)
# 2. Which of the two areas has the larger fluctuations over the whole slice?
# 3. Bonus: make the same selection for `"SSs"` **without** xarray, using NumPy
#    indexing on `lfp.values` and the `area` array. Which version would you
#    rather read in six months?

# %% tags=["solution"]
by_area = lfp.groupby("area").mean()
window = slice(stim_on[90] - 0.5, stim_on[90] + 1)

fig, ax = plt.subplots(figsize=(9, 3.5))
by_area.sel(area=["SSs", "SSp-n"], time=window).plot.line(x="time", hue="area", ax=ax)
ax.axvline(stim_on[90], color="grey", linestyle="--")
ax.set_title("Mean LFP around trial 90's stimulus")
print(by_area.sel(area=["SSs", "SSp-n"]).std(dim="time").round(1))

# %% tags=["solution"]
# Bonus: the NumPy way. You need to know the axis order and turn times into sample numbers yourself.
# A label slice includes both ends, so round the start up and the end down, then add 1.
first = int(np.ceil((stim_on[90] - 0.5 - lfp_start) * sampling_rate))
last = int(np.floor((stim_on[90] + 1 - lfp_start) * sampling_rate)) + 1
ss_numpy = lfp.values[first:last, area == "SSs"].mean(axis=1)
np.allclose(ss_numpy, by_area.sel(area="SSs", time=window).values)

# %% [markdown] tags=["answer"]
# `SSs` swings about two and a half times as far as `SSp-n` (a standard
# deviation of about 170 against 65 µV). The NumPy version works, but it depends on knowing that time is
# axis 0, on converting seconds to sample numbers by hand (and getting the
# rounding right), and on keeping `area` lined up with the columns. The xarray
# line says what it means.
#
# **Where next:** the [xarray tutorial](https://tutorial.xarray.dev), and
# `.chunk()` or `xr.open_dataset(..., chunks={})`, which hand the computation to
# **dask** (the next notebook).
