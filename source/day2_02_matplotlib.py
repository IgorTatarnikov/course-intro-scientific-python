# %% [markdown]
# # Day 2 · Matplotlib
#
# **Block:** Matplotlib (25 min, including slides and live coding)
#
# 1. **Simple plot** (8 min): build a figure of two LFP traces step by step,
#    then save it
# 2. **The whole recording as an image** (8 min): after SPL Exercise 31, fix a
#    first attempt that shows the data badly
# 3. **Stretch:** polish the first figure, a scatter of the neurons along the
#    probe, spike counts as an image, the population response, multiple
#    subplots, the LFP's envelope, and more SPL plot types
#
# You will often get a first version of a plot from an LLM or a colleague's
# script. The skills here are the ones you need to **check and fix** such a
# plot: know which object each line changes, and look hard at the axes, the
# units and the colours.
#
# We plot the recording from Day 1 (IBL, via the DANDI archive). The exercises
# follow [Scientific Python Lectures, Matplotlib:
# plotting](https://lectures.scientific-python.org/intro/matplotlib/index.html)
# (CC BY 4.0), which builds the same figures from sines and random numbers. We
# use the object-oriented style (`fig, ax = plt.subplots()`, as you did
# yesterday to check your results) throughout; SPL
# mostly uses the `plt.` functions, and the [Matplotlib
# docs](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html)
# explain the difference.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("../data/ibl_session.h5") as f:
    lfp_uv = f["lfp/data"][:] * f["lfp"].attrs["uV_per_count"]
    sampling_rate = f["lfp"].attrs["sampling_rate_Hz"]
    lfp_start = f["lfp"].attrs["start_time_s"]
    area = f["electrodes/area"][:].astype(str)
    depth_um = f["electrodes/depth_um"][:]
    stim_on = f["trials/stim_on_s"][:]
    contrast = f["trials/contrast"][:]
    rt = f["trials/response_time_s"][:]
    correct = f["trials/correct"][:]
    spike_counts = f["spike_counts/data"][:]
    spike_unit = f["spike_counts/unit"][:]
    bin_start = f["spike_counts/bin_start_s"][:]
    unit_depth = f["units/depth_um"][:]
    unit_rate = f["units/firing_rate_Hz"][:]
    unit_amplitude = f["units/amplitude_uV"][:]

time = np.arange(lfp_uv.shape[0]) / sampling_rate  # seconds into the slice

# %% [markdown]
# ## 1. Simple plot
#
# We want the LFP from two sites on one plot: channel 100, 1 mm above the probe
# tip in the visceral area (`VISC`), and channel 300, 3 mm up in primary
# somatosensory cortex (`SSp-n`). We start from the defaults and improve the
# figure step by step. Each step changes the same `fig` and `ax`; end each cell
# with `fig` so Jupyter shows the updated figure.

# %%
deep = lfp_uv[:, 100]
superficial = lfp_uv[:, 300]

# %% [markdown]
# **Step 1: defaults.** Create a figure and axes with `plt.subplots`, and plot
# `deep` and `superficial` against `time`.

# %% tags=["solution"]
fig, ax = plt.subplots()
ax.plot(time, deep)
ax.plot(time, superficial);

# %% [markdown]
# **Step 2: colours, line widths and a legend.** Make a new figure of size
# 10 x 4 inches (`figsize`). Plot `deep` in blue and `superficial` in red, both
# with a line width of 1, with `label="VISC, 1.0 mm"` and `label="SSp-n, 3.0
# mm"`. Add a legend in the upper left corner.

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, deep, color="blue", linewidth=1, label="VISC, 1.0 mm")
ax.plot(time, superficial, color="red", linewidth=1, label="SSp-n, 3.0 mm")
ax.legend(loc="upper left", frameon=False);

# %% [markdown]
# **Step 3: ticks and tick labels.** Three stimuli appeared during this slice:
# trials 88, 89 and 90. Put x ticks only at their times (`stim_on[88:91] -
# lfp_start` seconds into the slice) and label them `"trial 88"` and so on. Put
# y ticks at -400, 0 and 400 and label them with units, such as `"-400 µV"`.
# (`ax.set_xticks` takes both positions and `labels=`.)

# %% tags=["solution"]
stim_times = stim_on[88:91] - lfp_start
ax.set_xticks(stim_times, labels=["trial 88", "trial 89", "trial 90"])
ax.set_yticks([-400, 0, 400], labels=["-400 µV", "0", "400 µV"])
fig

# %% [markdown]
# **Step 4: save it.** Save the figure as `lfp_traces.png` at 150 dpi and as
# `lfp_traces.pdf`. Open both files. What happens when you zoom in on each?

# %% tags=["solution"]
fig.savefig("lfp_traces.png", dpi=150, bbox_inches="tight")
fig.savefig("lfp_traces.pdf", bbox_inches="tight")

# %% [markdown] tags=["solution"]
# The PNG is a grid of pixels and goes blocky when you zoom in. The PDF is a
# **vector** format and stays sharp, which is what journals usually want for
# line plots. Use `dpi=300` or more for raster images in print. (With very
# many points, as in a long recording, a vector file can get large and slow to
# open; then a high-dpi PNG is the better choice.)

# %% [markdown]
# ### Optional: polish the figure
#
# Skip to part 2 if you are short of time. These steps keep changing the same
# `fig` and `ax`.

# %% [markdown]
# **Limits.** Make the x axis run exactly from the first to the last
# time point, and the y axis from -1.2 to 1.2 times the largest absolute value in
# either trace (`ax.set_xlim`, `ax.set_ylim`), so 0 µV is in the middle.

# %% tags=["solution"]
ax.set_xlim(time.min(), time.max())
largest = max(np.abs(deep).max(), np.abs(superficial).max())
ax.set_ylim(-1.2 * largest, 1.2 * largest)
fig

# %% [markdown]
# **Spines.** Spines are the lines around the plotting area. Hide the top
# and right ones, and move the bottom one 10 points down, away from the traces
# (`ax.spines["bottom"].set_position(("outward", 10))`). Then draw a thin black
# horizontal line at 0 µV with `ax.axhline`.

# %% tags=["solution"]
ax.spines[["top", "right"]].set_visible(False)
ax.spines["bottom"].set_position(("outward", 10))
ax.axhline(0, color="black", linewidth=0.5)
fig

# %% [markdown]
# **Annotate.** Draw a
# dashed grey vertical line at each stimulus (`ax.axvline`). Label trial 89's
# with `ax.annotate`, for example "stimulus: 25% contrast, left" with an arrow
# pointing at the line. See the [annotation
# guide](https://matplotlib.org/stable/users/explain/text/annotations.html).

# %% tags=["solution"]
for t in stim_times:
    ax.axvline(t, color="grey", linewidth=1, linestyle="--")
ax.annotate(
    "stimulus: 25% contrast, left",
    xy=(stim_times[1], 0.8 * largest),
    xycoords="data",
    xytext=(30, 10),
    textcoords="offset points",
    fontsize=12,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
fig

# %% [markdown]
# Save the polished figure again, over the old files.

# %% tags=["solution"]
fig.savefig("lfp_traces.png", dpi=150, bbox_inches="tight")
fig.savefig("lfp_traces.pdf", bbox_inches="tight")

# %% [markdown]
# ## 2. The whole recording as an image (after SPL Exercise 31)
#
# The whole recording is a 2-D array, so it can be shown as an image: one row of
# pixels per channel, one column per time point. The first attempt below is not
# much use. Fix it, paying attention to:
#
# * the **aspect** ratio: 5000 time points by 384 channels makes a thin strip
#   (`aspect="auto"` fills the axes instead)
# * the **origin**: channel 0 is the **deepest** site, so it belongs at the
#   bottom (`origin="lower"`)
# * the **extent**: label the axes in seconds and micrometres instead of sample
#   and channel numbers (`extent=[left, right, bottom, top]`)
# * the **colormap**: a diverging map such as `"RdBu_r"`, centred on 0 µV with
#   `vmin` and `vmax`, and a **colorbar**

# %%
fig, ax = plt.subplots()
ax.imshow(lfp_uv.T);

# %% tags=["solution"]
limit = np.percentile(np.abs(lfp_uv), 99)
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    lfp_uv.T,
    aspect="auto",
    origin="lower",
    extent=[time[0], time[-1], depth_um[0], depth_um[-1]],
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    interpolation="nearest",
)
fig.colorbar(im, ax=ax, label="LFP (µV)")
ax.set_xlabel("time (s)")
ax.set_ylabel("height above probe tip (µm)");

# %% [markdown] tags=["solution"]
# `origin="lower"` puts row 0 at the bottom, like a graph, which here also matches
# the brain: the tip of the probe is the deepest point. The default,
# `origin="upper"`, puts row 0 at the top, like a matrix or a photograph; images
# from microscopes and cameras usually want the default. Clipping the colour
# scale at the 99th percentile stops a few large values from washing out the
# rest. A diverging colormap centred on 0 makes the sign of the signal easy to
# read: the broad vertical bands are slow waves that reach most of the probe at
# once.

# %% [markdown]
# ## 3. Stretch
#
# Work through any of these, in any order.

# %% [markdown]
# ### Scatter (after SPL Exercise 28)
#
# Each of the 481 units sits at some height on the probe. Starting from the code
# below, make a scatter plot of where the units are and what they look like,
# paying attention to marker **size**, **colour** and **transparency**:
#
# * Make each marker's size and colour show the unit's firing rate, with a
#   colorbar for the colour.
# * Make the markers half-transparent, so you can see where they pile up.
# * Put the amplitude on a log scale (`ax.set_xscale("log")`), and label both
#   axes with their units.

# %%
fig, ax = plt.subplots()
ax.scatter(unit_amplitude, unit_depth);

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(5, 7))
points = ax.scatter(
    unit_amplitude, unit_depth, s=5 * unit_rate, c=unit_rate, alpha=0.5, cmap="viridis"
)
fig.colorbar(points, ax=ax, label="firing rate (spikes/s)")
ax.set_xscale("log")
ax.set_xlabel("median spike amplitude (µV)")
ax.set_ylabel("height above probe tip (µm)");

# %% [markdown]
# ### Spike counts as an image
#
# `psth[unit, bin]` is each unit's mean spike count in each 50 ms bin, averaged
# over all 533 trials. It is a small image: one row per unit.

# %%
psth = spike_counts.mean(axis=1)
psth.shape

# %% [markdown]
# **a.** Display `psth` with `ax.imshow` and a colorbar. Only a few rows show
# anything. Why?

# %% tags=["solution"]
fig, ax = plt.subplots()
im = ax.imshow(psth, aspect="auto")
fig.colorbar(im, ax=ax, label="mean spikes per bin");

# %% [markdown] tags=["answer"]
# A few units fire much faster than the rest. The colour scale stretches to fit
# them, so the slower units all look dark.

# %% [markdown]
# **b. Normalise.** Make `psth_scaled` by dividing every row by its own
# maximum, so each unit runs from 0 to 1 (broadcasting, with `keepdims=True`).
# Display it.

# %% tags=["solution"]
psth_scaled = psth / psth.max(axis=1, keepdims=True)
fig, ax = plt.subplots()
im = ax.imshow(psth_scaled, aspect="auto")
fig.colorbar(im, ax=ax);

# %% [markdown]
# **c. Crop and sort.** Make `window` from `psth_scaled`, keeping only the bins
# from 0.5 s before to 0.5 s after the stimulus (the first 20). Then reorder its
# rows by the depth of each unit, deepest first. (Hint: `unit_depth[spike_unit]`
# gives the depth of each row; then `argsort` and fancy indexing.)

# %% tags=["solution"]
window = psth_scaled[:, :20]
order = unit_depth[spike_unit].argsort()
window = window[order]
window.shape

# %% [markdown]
# **d. Label it.** Display `window` with `origin="lower"` (deepest unit at the
# bottom), an `extent` that puts time in seconds on the x axis (the bins run
# from -0.5 to 0.5 s), and a dashed white vertical line at the stimulus, time 0.

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    window,
    aspect="auto",
    origin="lower",
    extent=[bin_start[0], bin_start[19] + 0.05, 0, len(window)],
    interpolation="nearest",
)
ax.axvline(0, color="white", linestyle="--")
fig.colorbar(im, ax=ax, label="spike count (fraction of the unit's maximum)")
ax.set_xlabel("time from stimulus (s)")
ax.set_ylabel("unit, deepest first");

# %% [markdown]
# ### The population response (part 6 of the data statistics in Day 1's NumPy)
#
# Plot the mean firing rate of all 51 units (in spikes per second: divide the
# mean count by the 0.05 s bin width) against time from the stimulus, once for
# **correct** trials and once for **wrong** ones, on the same axes. Mark the
# stimulus with a vertical line, and put the number of trials behind each line
# in its legend label.

# %% tags=["solution"]
rate_correct = spike_counts[:, correct].mean(axis=(0, 1)) / 0.05
rate_wrong = spike_counts[:, ~correct].mean(axis=(0, 1)) / 0.05
bin_centres = bin_start + 0.025

fig, ax = plt.subplots()
ax.plot(bin_centres, rate_correct, label=f"correct ({correct.sum()} trials)")
ax.plot(bin_centres, rate_wrong, label=f"wrong ({(~correct).sum()} trials)")
ax.axvline(0, color="grey", linestyle="--")
ax.set_xlabel("time from stimulus (s)")
ax.set_ylabel("mean firing rate (spikes/s)")
ax.legend();

# %% [markdown] tags=["solution"]
# Both lines rise after the stimulus; the one for wrong trials is noisier,
# because it averages over far fewer trials. An f-string in `label=` keeps the
# trial counts in the legend, so nobody has to guess them.

# %% [markdown]
# ### Multiple subplots (SPL Exercise 35)
#
# Reproduce the [SPL multi-plot layout](https://lectures.scientific-python.org/intro/matplotlib/index.html#multi-plots):
# one wide panel on top, three small panels below. SPL uses `plt.subplot`. Rebuild it
# with `plt.subplot_mosaic`, as in the Handbook's [Multiple
# Subplots](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
# chapter, which shows `GridSpec`, the older equivalent. Then put a plot of your
# choice of this recording in each panel.

# %% tags=["solution"]
fig, axd = plt.subplot_mosaic([["top", "top", "top"], ["a", "b", "c"]], figsize=(9, 5), layout="constrained")
axd["top"].plot(time, deep, linewidth=0.5, label="VISC")
axd["top"].plot(time, superficial, linewidth=0.5, label="SSp-n")
axd["top"].set_xlabel("time (s)")
axd["top"].legend()
axd["a"].hist(rt[rt < 3], bins=30)
axd["a"].set_xlabel("response time (s)")
axd["b"].scatter(unit_amplitude, unit_depth, s=5, alpha=0.5)
axd["b"].set_xlabel("amplitude (µV)")
axd["c"].imshow(lfp_uv[::10].T, aspect="auto", origin="lower", cmap="RdBu_r")
axd["c"].set_axis_off()

# %% [markdown]
# ### The envelope of the LFP
#
# Plot the **minimum** and the **maximum** LFP across all channels at each time
# point, on one set of axes, with a legend. Then shade the area between them with
# `ax.fill_between` (SPL Exercise 27). When is the spread largest?

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(10, 3))
lowest, highest = lfp_uv.min(axis=1), lfp_uv.max(axis=1)
ax.plot(time, lowest, linewidth=0.5, label="min over channels")
ax.plot(time, highest, linewidth=0.5, label="max over channels")
ax.fill_between(time, lowest, highest, alpha=0.3)
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (µV)")
ax.legend();

# %% [markdown]
# ### More plot types
#
# SPL Exercises 27 (`fill_between`), 29 (bar labels), 30 (contours), 32 (pie),
# 33 (quiver) and 36 (polar) in [Other Types of
# Plots](https://lectures.scientific-python.org/intro/matplotlib/index.html#other-types-of-plots-examples-and-exercises).
# Skip Exercise 37: its starter code uses `Axes3D(fig)`, which current Matplotlib
# no longer supports that way. Use `fig.add_subplot(projection="3d")` instead.
#
# Reference: the Handbook's chapters on [line
# plots](https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html),
# [error bars](https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html)
# and [colorbars](https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html).
