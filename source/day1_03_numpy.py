# %% [markdown]
# # Day 1 · NumPy
#
# **Block:** NumPy (80 min, including slides and live coding)
#
# The block alternates slides, live coding and exercises. Each part below
# matches one "Your turn" slide:
#
# 1. **Indexing** (12 min): time windows from chosen channels, and trials from
#    3-D spike counts
# 2. **Aggregations and broadcasting** (15 min): summaries along axes,
#    re-referencing and baseline correction
# 3. **Masks and vectorising** (18 min): select trials and channels by
#    condition, time a loop against NumPy, then the mouse's psychometric curve
#    without a single loop
# 4. **Check your work with a plot** (5 min)
# 5. **Stretch**, for anyone who finishes early: views and copies, in-place
#    changes, dtypes, more broadcasting, fancy indexing and data statistics
#
# The slides and live coding used 2-D tables of scores and a week of
# temperatures. The exercises use one real experiment, a recording from a
# mouse brain described below; only a few small SPL exercises use made-up
# arrays.
#
# Most tasks ask you to store a result in a named variable. Print it, and
# check its `.shape` and values against the array you started from. Part 4
# shows just enough Matplotlib to check a result by eye; plotting proper is on
# Day 2.
#
# The SPL exercises are adapted from [Scientific Python Lectures, NumPy
# exercises](https://lectures.scientific-python.org/intro/numpy/exercises.html)
# (CC BY 4.0). The rest covers the same ground as the Python Data Science
# Handbook chapters on [array
# basics](https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html),
# [broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html),
# [masks](https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html)
# and [fancy
# indexing](https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html).
# Read those for the full explanations.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=0)

# %% [markdown]
# ## The recording
#
# A mouse sits in front of a screen. A striped patch (the stimulus) appears on
# the left or the right, faint or strong, and the mouse turns a wheel to move it
# to the centre. A correct turn earns a drop of water. Meanwhile a
# **Neuropixels probe** records from the mouse's brain.
#
# The data come from the [International Brain Laboratory](https://www.internationalbrainlab.com/)
# and were downloaded from the [DANDI archive](https://dandiarchive.org/dandiset/000409)
# (CC BY 4.0; [doi:10.48324/dandi.000409/0.260309.1324](https://doi.org/10.48324/dandi.000409/0.260309.1324)).
# `../data/ibl_session.h5` holds a small slice of one session. It is
# an HDF5 file: a file of named arrays, arranged in groups like folders. The
# `h5py` package reads it, and `[:]` turns each one into a NumPy array:

# %%
with h5py.File("../data/ibl_session.h5") as f:
    # LFP: the voltage at every site, 500 times a second, for 10 s
    lfp = f["lfp/data"][:]
    uV_per_count = f["lfp"].attrs["uV_per_count"]
    sampling_rate = f["lfp"].attrs["sampling_rate_Hz"]
    lfp_start = f["lfp"].attrs["start_time_s"]
    # One entry per recording site (channel)
    area = f["electrodes/area"][:].astype(str)
    layer = f["electrodes/layer"][:].astype(str)
    x_um = f["electrodes/x_um"][:]
    depth_um = f["electrodes/depth_um"][:]
    # One entry per trial
    stim_on = f["trials/stim_on_s"][:]
    contrast = f["trials/contrast"][:]
    probability_left = f["trials/probability_left"][:]
    choice = f["trials/choice"][:]
    rt = f["trials/response_time_s"][:]
    correct = f["trials/correct"][:]
    # Spike counts of 51 neurons, in 50 ms bins around each stimulus
    spike_counts = f["spike_counts/data"][:]
    bin_start = f["spike_counts/bin_start_s"][:]
    # One entry per neuron (unit), including those not in spike_counts
    unit_channel = f["units/channel"][:]

# The raw LFP is in amplifier counts; this is the same data in microvolts
lfp_uv = lfp * uV_per_count

print(lfp.shape, lfp.dtype)
print(spike_counts.shape, spike_counts.dtype)
print(stim_on.shape, area.shape)

# %% [markdown]
# * **`lfp[time, channel]`** is the *local field potential*: the slow voltage
#   changes near each site, from 618 s into the session (`lfp_start`). Row `i` is
#   time `i / sampling_rate` seconds into the slice. The values are raw counts
#   from the amplifier; **`lfp_uv`** is the same in microvolts.
# * **Channels** are numbered from the **tip** of the probe: channel 0 is the
#   deepest site, and `depth_um` gives each site's height above the tip. `area`
#   says which brain area each site sits in (for example `SSp-n`, primary
#   somatosensory cortex, nose), and the top few sites (`void`) are above the
#   brain.
# * **`spike_counts[unit, trial, bin]`** is the number of spikes neuron `unit`
#   fired in time bin `bin` of trial `trial`. Bin `b` starts `bin_start[b]`
#   seconds after the stimulus, from -0.5 s (before it) to 0.95 s.
# * The **trial** arrays have one entry for each of the 533 trials: when the
#   stimulus appeared (`stim_on`, seconds into the session), its `contrast` (in
#   %, negative on the left, positive on the right), the block's
#   `probability_left` (explained in part 3), the mouse's `choice` (-1 left, 1
#   right), its response time `rt`, and whether it was `correct`.

# %%
print(area[:12])
print(depth_um[:12])

# %% [markdown]
# ## 1. Indexing
#
# Using **one indexing expression each** (no loops, no typing values in),
# make:
#
# 1. `deepest_trace`: the whole trace of channel 0, shape (5000,)
# 2. `one_value`: channel 100 at sample 1000. It should be 24.
# 3. `first_second`: every channel during the first second (500 samples),
#    shape (500, 384)
# 4. `sites_50_to_200`: channels 50 up to (not including) 200, from 2 s to 4 s,
#    shape (1000, 150)
# 5. `last_second`: every channel during the last second, shape (500, 384)
# 6. `every_other_site`: every other channel (0, 2, 4, ...), shape (5000, 192)
# 7. `top_first`: the whole recording with the channels in reverse order, so
#    that column 0 is the **top** of the probe
# 8. `downsampled`: every 5th sample of every channel (100 samples per second),
#    shape (1000, 384)
#
# Check the `.shape` of each result against the one given.

# %%
# BEGIN SOLUTION
deepest_trace = lfp[:, 0]
one_value = lfp[1000, 100]
first_second = lfp[:500]
sites_50_to_200 = lfp[1000:2000, 50:200]
last_second = lfp[-500:]
every_other_site = lfp[:, ::2]
top_first = lfp[:, ::-1]
downsampled = lfp[::5]
# END SOLUTION

# %% tags=["solution"]
print(deepest_trace.shape, one_value)
for result in [first_second, sites_50_to_200, last_second, every_other_site, top_first, downsampled]:
    print(result.shape)
print(area[[0, -1]], area[::-1][[0, -1]])  # top_first's first column is a void site

# %% [markdown]
# Now in `spike_counts`, which has three axes `[unit, trial, bin]`:
#
# 9. `one_unit_one_trial`: unit 3's counts in every bin of trial 10, shape (30,)
# 10. `after_stimulus`: every unit and trial, but only the bins from the
#     stimulus on (`bin_start` 0 and later, which is bin 10 onwards), shape
#     (51, 533, 20)
# 11. `first_bin`: the first bin of every trial for every unit, shape (51, 533)
#
# Finally, a time window that you have to **compute** rather than read off:
#
# 12. Trial 89's stimulus appeared at `stim_on[89]` seconds into the session,
#     which is somewhere in our LFP slice. Make `around_stimulus`: every channel
#     from 0.5 s before to 1 s after it, shape (750, 384). (Hint: work out the
#     sample number of the stimulus from `lfp_start` and `sampling_rate`, and
#     turn it into an `int`.)

# %% tags=["solution"]
one_unit_one_trial = spike_counts[3, 10]
after_stimulus = spike_counts[:, :, 10:]
first_bin = spike_counts[:, :, 0]
print(one_unit_one_trial.shape, after_stimulus.shape, first_bin.shape)

stim_sample = int((stim_on[89] - lfp_start) * sampling_rate)
around_stimulus = lfp[stim_sample - 250 : stim_sample + 500]
print(stim_sample, around_stimulus.shape)

# %% [markdown]
# ## 2. Aggregations and broadcasting
#
# ### 2a. Aggregations along more than one axis
#
# Back to `spike_counts[unit, trial, bin]`, shape (51, 533, 30). Make:
#
# 1. `total_per_unit`: the total number of spikes from each unit, over all
#    trials and bins (51 values). (Hint: `axis` also accepts a tuple of axes.)
# 2. `psth`: for each unit, the mean count in each time bin, averaged over
#    trials. Shape (51, 30). This is called a *peri-stimulus time histogram*.
# 3. `busiest_trial`: for each unit, the **number** of the trial in which it
#    fired the most spikes (51 values).
# 4. `overall_mean`: the mean count over the whole array (one number)
# 5. `population`: the mean over units **and** trials, one value per bin, in
#    spikes per second (divide by the 0.05 s bin width). Does the population
#    fire more after the stimulus (bin 10 onwards) than before?

# %% tags=["solution"]
total_per_unit = spike_counts.sum(axis=(1, 2))
psth = spike_counts.mean(axis=1)
busiest_trial = spike_counts.sum(axis=2).argmax(axis=1)
overall_mean = spike_counts.mean()
population = spike_counts.mean(axis=(0, 1)) / 0.05
print(total_per_unit, psth.shape, busiest_trial, overall_mean, sep="\n")
print(population.round(1))

# %% [markdown] tags=["answer"]
# Yes: about 3 spikes per second before the stimulus and about 4 after it,
# rising from bin 12 (0.1 s after the stimulus appears).

# %% [markdown]
# ### 2b. Broadcasting
#
# No loops in this section.
#
# 1. Noise that reaches every site at once (from the mouse moving, say) can be
#    removed by subtracting, at each time point, the median over all channels.
#    Make `lfp_car` (for *common average reference*) from `lfp_uv`, using
#    `np.median` with `keepdims=True`. Its shape should still be (5000, 384).
# 2. Make `psth_change`: each unit's `psth` minus that unit's mean over the
#    bins **before** the stimulus (the first 10 bins). Shape (51, 30); the first
#    10 values of each row should average 0.

# %% tags=["solution"]
lfp_car = lfp_uv - np.median(lfp_uv, axis=1, keepdims=True)  # (5000, 384) - (5000, 1)
baseline = psth[:, :10].mean(axis=1, keepdims=True)  # (51, 1)
psth_change = psth - baseline
print(lfp_car.shape, psth_change.shape)
print(psth_change[:, :10].mean(axis=1).round(10)[:5])

# %% [markdown]
# 3. The cell below should keep the spike counts of the **correct** trials and
#    set the others to 0, by multiplying by `correct` (one `True`/`False` per
#    trial), but it fails. Fix it.

# %% tags=["raises-exception"]
spike_counts * correct

# %% tags=["solution"]
# Shapes are compared from the right: (51, 533, 30) against (533,) lines up 30
# against 533. Making correct (533, 1) lines 533 up with the trial axis instead.
correct_only = spike_counts * correct[:, np.newaxis]
correct_only.shape

# %% [markdown]
# ## 3. Masks and vectorising
#
# ### 3a. Boolean masks
#
# `rt` holds the mouse's response time on each trial, in seconds from the
# stimulus to its choice, and `correct` says whether the choice was right.
# Some trials are odd: on a few the mouse was already turning the wheel, and on
# others it lost interest.

# %%
rt[:10].round(3), correct[:10]

# %% [markdown]
# Make:
#
# 1. `too_fast`: the response times below 0.15 s (faster than the mouse could
#    have seen the stimulus)
# 2. `n_correct`: the number of correct trials
# 3. `mean_rt_correct`: the mean response time of the **correct** trials only
# 4. `slow_errors`: the response times of trials that were **wrong** and slower
#    than 1 s
# 5. `rt_clean`: a **new** array in which every response time below 0.15 or
#    above 5 s is replaced by `np.nan`. `rt` must not change. Then compare
#    `rt.mean()` with `np.nanmean(rt_clean)`.

# %% tags=["solution"]
too_fast = rt[rt < 0.15]
n_correct = correct.sum()
mean_rt_correct = rt[correct].mean()
slow_errors = rt[~correct & (rt > 1)]
rt_clean = np.where((rt < 0.15) | (rt > 5), np.nan, rt)
print(too_fast, n_correct, mean_rt_correct, slow_errors.round(2), sep="\n")
print(rt.mean(), np.nanmean(rt_clean))

# %% [markdown]
# Masks work on channels too. `area` has one entry per channel, so a mask built
# from it picks out **columns** of `lfp_uv`. Make:
#
# 6. `ssp_lfp`: the LFP of every site in primary somatosensory cortex
#    (`"SSp-n"`), shape (5000, 128)
# 7. `in_brain`: the LFP of every site **except** those above the brain
#    (`"void"`), shape (5000, 378)

# %% tags=["solution"]
ssp_lfp = lfp_uv[:, area == "SSp-n"]
in_brain = lfp_uv[:, area != "void"]
print(ssp_lfp.shape, in_brain.shape)

# %% [markdown]
# ### 3b. Vectorised versus loops
#
# The **line length** of a signal is the sum of the absolute differences between
# consecutive samples. It is large when the signal is busy, and is a cheap
# measure of activity in seizure detection.
#
# `line_length_loop(signals)` below computes it with `for` loops over the
# channels and the samples. Write `line_length_numpy(signals)`, which does the
# same with no loop at all. Each takes an array of shape (time, channel) and
# returns one value per channel. Check that they agree on `first_2_s`, then time
# both with `%timeit`. How many times faster is NumPy? (Hint for the NumPy version: `signals[1:] -
# signals[:-1]` gives every difference at once, or use `np.diff`. See also the
# Handbook's [Profiling and Timing
# Code](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html).)

# %%
first_2_s = lfp_uv[:1000]  # the first 2 s, in microvolts


def line_length_loop(signals):
    n_time, n_channels = signals.shape
    result = np.zeros(n_channels)
    for channel in range(n_channels):
        total = 0.0
        for t in range(1, n_time):
            total += abs(signals[t, channel] - signals[t - 1, channel])
        result[channel] = total
    return result


def line_length_numpy(signals):
    # BEGIN SOLUTION
    return np.abs(signals[1:] - signals[:-1]).sum(axis=0)
    # END SOLUTION


# %% tags=["solution"]
np.allclose(line_length_loop(first_2_s), line_length_numpy(first_2_s))

# %% tags=["solution"]
# %timeit -n 1 -r 3 line_length_loop(first_2_s)
# %timeit -n 3 -r 3 line_length_numpy(first_2_s)

# %% [markdown] tags=["answer"]
# On a typical laptop the loop takes tens of milliseconds and NumPy under one:
# roughly 100 times faster. The loop pays for Python work at every one of the
# 384 000 samples (indexing, subtraction, `abs`); NumPy does each step once,
# over the whole array, in compiled code. If you catch yourself writing a `for`
# loop over the values of an array, there is almost always a NumPy way.

# %% [markdown]
# ### 3c. The psychometric curve, without a loop
#
# Does the mouse do the task? The **psychometric curve** is the fraction of
# trials on which it chose **right**, at each contrast. There are 9 contrasts:

# %%
levels = np.unique(contrast)
levels

# %% [markdown]
# The obvious way is a loop over `levels`, picking out the trials at each one.
# Do it with a mask and broadcasting instead. `y` is `True` where the mouse
# chose right:

# %%
y = choice == 1

# %% [markdown]
#
# 1. `at_level`: a (533, 9) boolean array, `True` where trial `i` had contrast
#    `levels[j]`. (Hint: compare `contrast` with `levels`. Which of them needs
#    a new axis so that the result is (533, 9)?)
# 2. `n_at_level`: the number of trials at each contrast (9 values).
# 3. `frac_right`: the fraction of rightward choices at each contrast. (Hint:
#    `at_level & y[:, np.newaxis]` is `True` where a trial was at that contrast
#    **and** the mouse chose right. Count, then divide.)
#
# The task also has **blocks** of trials. The session starts with 90 unbiased
# trials (`probability_left` 0.5). Then, for a while, the stimulus appears on the
# left 80% of the time (0.8), then on the right 80% of the time (0.2), and so on.
#
# 4. Make `frac_right_rb` and `frac_right_lb`: the same curve for the
#    right-block trials (`probability_left == 0.2`) and the left-block trials
#    (0.8) only. (Hint: select the rows of `at_level` and `y` with a mask
#    first.) Compare the two at contrast 0, where there is nothing to see. Has
#    the mouse learned the blocks?

# %% tags=["solution"]
at_level = contrast[:, np.newaxis] == levels  # (533, 1) == (9,) -> (533, 9)
n_at_level = at_level.sum(axis=0)
frac_right = (at_level & y[:, np.newaxis]).sum(axis=0) / n_at_level
print(n_at_level)
print(frac_right.round(2))

# %% tags=["solution"]
right_block = probability_left == 0.2
left_block = probability_left == 0.8
frac_right_rb = (at_level[right_block] & y[right_block, np.newaxis]).sum(axis=0) / at_level[right_block].sum(axis=0)
frac_right_lb = (at_level[left_block] & y[left_block, np.newaxis]).sum(axis=0) / at_level[left_block].sum(axis=0)
print(frac_right_rb.round(2))
print(frac_right_lb.round(2))

# %% [markdown] tags=["answer"]
# At strong contrasts the mouse is nearly always right, whatever the block. At
# contrast 0 it chooses right about half the time in right blocks but under a
# third of the time in left blocks: when it cannot see the stimulus, it guesses
# the side that has been more likely lately. It has learned the blocks and uses
# them as a prior. (`(at_level & y[:, np.newaxis]).mean(axis=0)` would be wrong:
# it divides by all 533 trials, not by the trials at each contrast.)

# %% [markdown]
# ## 4. Check your work with a plot
#
# A column of numbers is hard to judge; a plot is easy. Here is all the
# Matplotlib you need today. Day 2 covers it properly.

# %%
time = np.arange(500) / sampling_rate  # the first second, in seconds

fig, ax = plt.subplots()
ax.plot(time, lfp_uv[:500, 100], label="channel 100")
ax.plot(time, lfp_uv[:500, 300], label="channel 300")
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (µV)")
ax.legend();

# %% [markdown]
# * `fig, ax = plt.subplots()` makes a figure with one set of axes
# * `ax.plot(x, y)` draws a line; `ax.plot(x, y, "o-")` adds a marker at each
#   point
# * `ax.set_xlabel`, `ax.set_ylabel` and `ax.legend()` (which uses each line's
#   `label=`) label it
#
# Plot `frac_right_rb` and `frac_right_lb` against `levels` on one set of axes,
# with markers, labelled axes and a legend. Does the plot agree with what you
# concluded from the numbers?

# %% tags=["solution"]
fig, ax = plt.subplots()
ax.plot(levels, frac_right_rb, "o-", label="right blocks (p left = 0.2)")
ax.plot(levels, frac_right_lb, "o-", label="left blocks (p left = 0.8)")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("fraction of rightward choices")
ax.legend();

# %% [markdown]
# ## 5. Stretch
#
# For anyone who finishes early, in any order.
#
# ### 5a. Build an array without typing it in (SPL Exercise 22, part 1)
#
# Form the 2-D array below **without typing it in explicitly**, and call it
# `a`:
#
# ```
# [[1,  6, 11],
#  [2,  7, 12],
#  [3,  8, 13],
#  [4,  9, 14],
#  [5, 10, 15]]
# ```
#
# Then make `rows_2_and_4`, a new array containing its 2nd and 4th rows.

# %% tags=["solution"]
a = np.arange(1, 16).reshape(3, 5).T
rows_2_and_4 = a[1::2]  # or a[[1, 3]]
print(a)
print(rows_2_and_4)

# %% [markdown]
# ### 5b. Views and copies
#
# 1. Make `channel_100`, the trace of channel 100. Its first 0.1 s (50 samples)
#    is an artefact: set them to 0 **in `channel_100`**. Then print
#    `lfp[:55, 100]`: did `lfp` change?
# 2. Reset `lfp` by reading it from the file again (the cell below). Now do the
#    same again, but this time **without changing `lfp`**.

# %% tags=["solution"]
# 1. A column of an array is a view, so lfp changes too.
channel_100 = lfp[:, 100]
channel_100[:50] = 0
print(lfp[:55, 100])

# %%
with h5py.File("../data/ibl_session.h5") as f:
    lfp = f["lfp/data"][:]

# %% tags=["solution"]
# 2. Copy first.
channel_100 = lfp[:, 100].copy()
channel_100[:50] = 0
print(lfp[:55, 100])

# %% [markdown]
# 3. Which of these share memory with `lfp`, and which are copies? Find out
#    with `np.shares_memory(lfp, ...)` rather than guessing: `lfp[:, 100]`,
#    `lfp[:, ::2]`, `lfp[:, [50, 200]]`, `lfp[lfp > 100]`, `lfp.T`,
#    `lfp.ravel()`, `lfp.flatten()`.

# %% tags=["solution"]
for name, candidate in [
    ("lfp[:, 100]", lfp[:, 100]),
    ("lfp[:, ::2]", lfp[:, ::2]),
    ("lfp[:, [50, 200]]", lfp[:, [50, 200]]),
    ("lfp[lfp > 100]", lfp[lfp > 100]),
    ("lfp.T", lfp.T),
    ("lfp.ravel()", lfp.ravel()),
    ("lfp.flatten()", lfp.flatten()),
]:
    print(f"{name:19} shares memory: {np.shares_memory(lfp, candidate)}")

# %% [markdown] tags=["answer"]
# Columns, slices (even with a step), `.T` and `ravel()` are views. Indexing with a
# list of integers (fancy indexing) or with a mask makes a copy, and so does
# `flatten()`, which always copies (`ravel()` copies only when it has to).

# %% [markdown]
# ### 5c. In place, or a new array?
#
# A colleague wrote this function to shift a signal so that its minimum is 0.
# `readings` are four LFP samples, in microvolts:

# %%
def remove_offset(signal):
    signal -= signal.min()
    return signal


readings = np.array([12.0, 15.5, 13.2, 18.1])
shifted = remove_offset(readings)
print(shifted)
print(readings)

# %% [markdown]
# 1. `readings` changed, although the function returned a result. Why?
# 2. Write `remove_offset_safe`, which returns the shifted signal but leaves its
#    input alone. Check it on a fresh `readings`.

# %% [markdown] tags=["answer"]
# Inside the function, `signal` is just another name for the caller's array, and
# `-=` changes that array in place. The function returns the same array it was
# given, so `shifted` and `readings` are one array with two names.

# %%
def remove_offset_safe(signal):
    # BEGIN SOLUTION
    return signal - signal.min()
    # END SOLUTION


readings = np.array([12.0, 15.5, 13.2, 18.1])

# %% tags=["solution"]
shifted = remove_offset_safe(readings)
print(shifted)
print(readings)

# %% [markdown]
# NumPy's random generator has the same split. `rng.permutation(x)` returns a
# shuffled copy and `rng.shuffle(x)` shuffles `x` itself. Shuffling trial
# numbers is how you build a *shuffle control*: is a neuron's response tied to
# the stimulus, or would any trial order do?
#
# 3. Make `shuffled_order`, a shuffled copy of `order`, leaving `order`
#    unchanged. Print `order` to check.
# 4. Now shuffle `order` itself, in place. What does `rng.shuffle` return?

# %%
order = np.arange(10)  # trial numbers

# %% tags=["solution"]
shuffled_order = rng.permutation(order)
print(order, shuffled_order)

# %% tags=["solution"]
result = rng.shuffle(order)
print(order, result)

# %% [markdown] tags=["answer"]
# `rng.shuffle` changes `order` in place and returns `None`, so
# `order = rng.shuffle(order)` would throw your data away.

# %% [markdown]
# ### 5d. dtypes
#
# `lfp` holds whole numbers from the amplifier, stored as `int16` (2 bytes
# each, from -32768 to 32767) to save space. That is why the first cell of this
# notebook made `lfp_uv` as a **new** array: converting `lfp` to microvolts in
# place fails.

# %% tags=["raises-exception"]
lfp *= uV_per_count

# %% [markdown]
# The power of a signal is its square. Squaring the raw counts goes wrong:

# %%
(lfp**2).min()  # a square can't be negative

# %% [markdown]
# 1. Why does the in-place conversion fail, and what is the dtype of `lfp_uv`?
# 2. Why is the square negative? (Hint: what is 225 squared, and what is the
#    largest `int16`?)
# 3. Make `lfp_power`, the square of every sample in microvolts squared, and
#    check that its minimum is not negative.

# %% tags=["solution"]
print(lfp_uv.dtype, lfp.dtype)
lfp_power = lfp_uv**2
print(lfp_power.min(), lfp_power.max())

# %% [markdown] tags=["answer"]
# An in-place operation has to keep the array's dtype, and the product of an
# `int16` and a float is a float, which cannot be stored back into `int16`.
# `lfp * uV_per_count` makes a new `float64` array instead. The largest value
# in `lfp` is 225, and 225² = 50625 does not fit in an `int16`, whose largest
# value is 32767. NumPy keeps the dtype of its inputs and does not warn you: the
# result wraps round to a negative number. Converting to float first (or to a
# wider integer such as `np.int32`) avoids it.

# %% [markdown]
# ### 5e. More broadcasting
#
# 1. The sites of a Neuropixels probe sit in a staggered pattern: `x_um` and
#    `depth_um` give each site's position in micrometres. Make `distances`, the
#    384 x 384 array where `distances[i, j]` is the distance between site `i`
#    and site `j`. (Hint: stack the positions into `positions`, shape (384, 2).
#    Give one copy the shape (384, 1, 2) and another (1, 384, 2). Their
#    difference has shape (384, 384, 2). Then square, sum over the last axis
#    and take the square root.) Check that the diagonal is 0, and look at
#    `distances[:4, :4]`.

# %% tags=["solution"]
positions = np.column_stack([x_um, depth_um])
offsets = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
distances = np.sqrt((offsets**2).sum(axis=-1))
print(distances.shape, np.diag(distances).max())
print(distances[:4, :4].round(1))

# %% [markdown]
# 2. **(SPL Exercise 22, part 2)** Divide each **column** of `a` element-wise
#    by `b`, so that row `i` is divided by `b[i]`. (Hint: `np.newaxis`.)

# %%
a = np.arange(25).reshape(5, 5)
b = np.array([1.0, 5, 10, 15, 20])

# %% tags=["solution"]
# a has shape (5, 5) and b has shape (5,). b[:, np.newaxis] has shape (5, 1),
# so b[i] divides every value in row i, which is element i of each column.
a / b[:, np.newaxis]

# %% [markdown] tags=["solution"]
# Note that `a / b` also runs without error, but it divides each **row** by `b`
# (it lines `b` up along the last axis). It is wrong, and NumPy will not warn you,
# so always check which axis broadcasting used.

# %% [markdown]
# ### 5f. More masks
#
# 1. `layer_5_somatosensory`: the LFP of the sites in layer 5 (`layer == "5"`)
#    of **either** somatosensory area, `"SSp-n"` or `"SSs"`. How many sites is
#    that? (Hint: `np.isin`, or two comparisons joined with `|`.)
# 2. `silent_trials`: for unit 0, the number of trials in which it fired **no
#    spikes at all**. (Hint: `np.all` with an `axis`.)

# %% tags=["solution"]
somatosensory = (area == "SSp-n") | (area == "SSs")  # or np.isin(area, ["SSp-n", "SSs"])
layer_5_somatosensory = lfp_uv[:, somatosensory & (layer == "5")]
silent_trials = np.all(spike_counts[0] == 0, axis=1).sum()
print(layer_5_somatosensory.shape, silent_trials)

# %% [markdown]
# ### 5g. Fancy indexing
#
# An integer array can index into a **lookup table**: every index is replaced
# by the matching entry.
#
# 1. `unit_channel` gives the channel on which each of the 481 neurons (units)
#    was largest. Make `unit_area`: the brain area of each unit, using `area`
#    as the lookup table. How many units are in `"SSp-n"`? How many are
#    in `"void"` sites, above the brain?

# %% tags=["solution"]
unit_area = area[unit_channel]
print(unit_area[:10])
print((unit_area == "SSp-n").sum(), (unit_area == "void").sum())

# %% [markdown]
# 2. Make `rt_sorted`, the response times sorted fastest first, and
#    `correct_sorted`, the `correct` values reordered **the same way** so that
#    they still belong to the same trials. (Hint: `argsort`.)

# %% tags=["solution"]
order = rt.argsort()
rt_sorted = rt[order]
correct_sorted = correct[order]
print(rt_sorted[:5].round(3), correct_sorted[:5])

# %% [markdown]
# 3. To draw the probe, we want one colour per channel, by area. `area_names`
#    lists the areas and `area_code` gives each channel's position in that list
#    (so `area_names[area_code]` is `area` again). `palette` gives one RGB
#    colour per area. Make `channel_colours`, an RGB colour for every channel,
#    using **one** indexing expression. What shape is it, and why?
#
# Then, in place:
#
# 4. Channels 12, 57 and 300 are known to be noisy. Set them to 0 in
#    `lfp_uv`, using one integer list as the index.

# %%
area_names, area_code = np.unique(area, return_inverse=True)
palette = rng.integers(0, 256, size=(area_names.size, 3))
area_names, area_code[:12]

# %% tags=["solution"]
channel_colours = palette[area_code]
print(channel_colours.shape)
print(channel_colours[:3])

# %% [markdown] tags=["answer"]
# `(384, 3)`: every entry of the (384,) `area_code` array is replaced by a row of
# `palette`, which has 3 values. Fancy indexing gives the shape of the index
# array, followed by the remaining axes of the array being indexed.

# %% tags=["solution"]
bad_channels = [12, 57, 300]
lfp_uv[:, bad_channels] = 0
print(lfp_uv[:3, [11, 12, 13]])

# %% [markdown]
# 5. **(SPL Exercise 22, part 3; harder)** `r` is a 10 x 3 array of random
#    numbers in [0, 1]. Make `closest`: for each row, the number closest to
#    0.5.
#
#    * Use `abs` and `argmin` to find the column `j` closest to 0.5 in each row.
#    * Use fancy indexing to extract the numbers. (Hint: `r[i, j]`, where the
#      array `i` contains the row numbers that go with `j`.)

# %%
r = rng.random((10, 3))

# %% tags=["solution"]
j = np.abs(r - 0.5).argmin(axis=1)
i = np.arange(r.shape[0])
closest = r[i, j]
print(r.round(2))
print(closest.round(2))

# %% [markdown]
# ### 5h. Data statistics (after SPL Exercise 23)
#
# Time to put it together. `spike_unit` gives the unit number of each row of
# `spike_counts`, the same numbering as in `unit_channel`.

# %%
with h5py.File("../data/ibl_session.h5") as f:
    spike_unit = f["spike_counts/unit"][:]
spike_unit[:10]

# %% [markdown]
# Compute and print, **without any for-loops**:
#
# 1. The mean and standard deviation of the response time, for correct and for
#    wrong trials. What does the difference tell you about the mouse?
# 2. For each unit, the time (from `bin_start`) of the bin where its
#    trial-averaged count is highest. (Hint: `argmax` of `psth`, then fancy
#    indexing of `bin_start`.)
# 3. For each trial, the **unit number** that fired the most spikes over the
#    whole trial. Is it the same unit every time? (Hint: `argmax` and fancy
#    indexing of `spike_unit`.)
# 4. The trials in which **any** unit fired more than 10 spikes in a single
#    50 ms bin. (Hint: comparisons, and `np.any` over two axes.)
# 5. The unit numbers of the two units that fired the most spikes in total.
#    (Hint: `argsort` and fancy indexing.)
#
# (Part 6, the population response on correct and wrong trials, is a stretch
# exercise in Day 2's Matplotlib notebook.)

# %% tags=["solution"]
# 1.
print(f"Correct: {rt[correct].mean():.2f} +/- {rt[correct].std():.2f} s")
print(f"Wrong:   {rt[~correct].mean():.2f} +/- {rt[~correct].std():.2f} s")

# %% [markdown] tags=["answer"]
# Wrong trials are about three times slower on average, and far more variable.
# Many of them are trials where the mouse was not paying attention. The
# standard deviations are larger than the means because a few very slow trials
# (over 10 s) dominate; the median is a better summary here.

# %% tags=["solution"]
# 2.
psth = spike_counts.mean(axis=1)
print("Peak time per unit (s):", bin_start[psth.argmax(axis=1)])

# %% tags=["solution"]
# 3.
spikes_per_trial = spike_counts.sum(axis=2)  # (unit, trial)
busiest_unit = spike_unit[spikes_per_trial.argmax(axis=0)]
print(busiest_unit[:20])
print("Units that were ever the busiest:", np.unique(busiest_unit))

# %% [markdown] tags=["answer"]
# Almost always the same unit: one neuron fires far faster than the others, so it
# wins on nearly every trial. Comparing raw counts between neurons mostly tells
# you which one fires fastest overall, which is why analyses usually compare each
# neuron with its own baseline (as `psth_change` did in part 2).

# %% tags=["solution"]
# 4.
print("Trials with a bin above 10:", np.flatnonzero(np.any(spike_counts > 10, axis=(0, 2))))

# %% tags=["solution"]
# 5. argsort sorts ascending, so the last two are the largest.
print("Two busiest units:", spike_unit[spikes_per_trial.sum(axis=1).argsort()[-2:][::-1]])
