# %% [markdown]
# # Day 1 · NumPy
#
# **Block:** NumPy (75 min, including slides and live coding)
#
# The block alternates slides, live coding and exercises. Each part below
# matches one "Your turn" slide:
#
# 1. **Indexing, views and in-place changes** (15 min)
# 2. **Aggregations and broadcasting** (12 min)
# 3. **Masks, fancy indexing and vectorising** (20 min), finishing with the
#    hare and lynx data (SPL Exercise 23)
# 4. **Stretch:** crude integral approximations (SPL Exercise 24)
# 5. **Stretch:** Markov chain (SPL Exercise 26)
#
# The slides and live coding used 2-D tables of scores and a week of
# temperatures. The exercises practise the same ideas on different data:
# a small video, spike counts, reaction times and an animal's path.
#
# Most tasks ask you to store a result in a named variable. Print it, and
# check it by eye against the array you started from. There is no plotting
# yet; plotting starts on Day 2.
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
import numpy as np

rng = np.random.default_rng(seed=0)

# %% [markdown]
# ## 1. Indexing, views and in-place changes
#
# ### 1a. Fetching parts of a 3-D array
#
# On the slides the arrays were 2-D. Here `movie` is a tiny 3-D array: 3 frames
# of a 4 x 5 pixel video, indexed as `movie[frame, row, column]`.

# %%
movie = np.arange(60).reshape(3, 4, 5)
movie

# %% [markdown]
# Using **one indexing expression each** (no loops, no typing values in),
# make:
#
# 1. `first_frame`: the whole first frame, shape (4, 5)
# 2. `pixel_value`: frame 1, row 2, column 3 (counting from 0). It should be 33.
# 3. `pixel_trace`: the pixel at row 2, column 3 in **every** frame:
#    `[13, 33, 53]`
# 4. `last_two_frames`: the last two frames, shape (2, 4, 5)
# 5. `top_rows`: the top row of every frame, shape (3, 5)
# 6. `cropped`: rows 1 to 2 and columns 1 to 3 of every frame, shape (3, 2, 3)
# 7. `mirrored`: every frame flipped left to right (columns reversed)
# 8. `downsampled`: every other row and every other column of every frame,
#    shape (3, 2, 3)
#
# Check the `.shape` of each result against the one given.

# %%
# BEGIN SOLUTION
first_frame = movie[0]
pixel_value = movie[1, 2, 3]
pixel_trace = movie[:, 2, 3]
last_two_frames = movie[-2:]
top_rows = movie[:, 0, :]
cropped = movie[:, 1:3, 1:4]
mirrored = movie[:, :, ::-1]
downsampled = movie[:, ::2, ::2]
# END SOLUTION

# %% tags=["solution"]
print(first_frame, pixel_value, pixel_trace, sep="\n\n")
for result in [last_two_frames, top_rows, cropped, mirrored, downsampled]:
    print(result.shape)
print(mirrored[0], downsampled[0], sep="\n\n")

# %% [markdown]
# ### 1b. Build an array without typing it in (SPL Exercise 22, part 1)
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
# ### 1c. Views and copies
#
# `recording` holds 8 time points from each of 3 electrodes (one row per
# electrode).
#
# 1. Make `channel_1`, the second electrode's row. Its first two samples are an
#    artefact: set them to 0 **in `channel_1`**. Print `recording`: did it change?
# 2. Reset `recording`. Now do the same again, but this time **without changing
#    `recording`**.

# %%
recording = np.arange(24).reshape(3, 8)

# %% tags=["solution"]
# 1. A row of an array is a view, so recording changes too.
channel_1 = recording[1]
channel_1[:2] = 0
print(recording)

# %% tags=["solution"]
# 2. Copy first.
recording = np.arange(24).reshape(3, 8)
channel_1 = recording[1].copy()
channel_1[:2] = 0
print(recording)

# %% [markdown]
# 3. Which of these share memory with `recording`, and which are copies? Find
#    out with `np.shares_memory(recording, ...)` rather than guessing:
#    `recording[1]`, `recording[:, ::2]`, `recording[[0, 2]]`,
#    `recording[recording > 10]`, `recording.T`, `recording.ravel()`,
#    `recording.flatten()`.

# %% tags=["solution"]
for name, candidate in [
    ("recording[1]", recording[1]),
    ("recording[:, ::2]", recording[:, ::2]),
    ("recording[[0, 2]]", recording[[0, 2]]),
    ("recording[recording > 10]", recording[recording > 10]),
    ("recording.T", recording.T),
    ("recording.ravel()", recording.ravel()),
    ("recording.flatten()", recording.flatten()),
]:
    print(f"{name:27} shares memory: {np.shares_memory(recording, candidate)}")

# %% [markdown] tags=["answer"]
# Rows, slices (even with a step), `.T` and `ravel()` are views. Indexing with a
# list of integers (fancy indexing) or with a mask makes a copy, and so does
# `flatten()`, which always copies (`ravel()` copies only when it has to).

# %% [markdown]
# ### 1d. In place, or a new array?
#
# A colleague wrote this function to shift a signal so that its minimum is 0:

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
# shuffled copy and `rng.shuffle(x)` shuffles `x` itself.
#
# 3. Make `shuffled_order`, a shuffled copy of `order`, leaving `order`
#    unchanged. Print `order` to check.
# 4. Now shuffle `order` itself, in place. What does `rng.shuffle` return?

# %%
order = np.arange(10)

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
# ### 1e. dtypes
#
# Adding 0.5 to an integer array in place fails:

# %% tags=["raises-exception"]
counts = np.array([3, 7, 2])
counts += 0.5

# %% [markdown]
# 1. Make `counts_half`, a **float** array holding `counts + 0.5`.
#
# `frame_a` and `frame_b` are the same three pixels in two video frames,
# stored as `uint8` (0 to 255), as images usually are. Subtracting them to see
# what changed goes wrong:

# %%
frame_a = np.array([12, 200, 250], dtype=np.uint8)
frame_b = np.array([10, 210, 250], dtype=np.uint8)
frame_b - frame_a  # 10 - 12 should be -2, not 254

# %% [markdown]
# 2. Make `change`, the correct difference `frame_b - frame_a`, `[-2, 10, 0]`.
#    `frame_a` and `frame_b` must not change. (Hint: `.astype()` to a type that
#    can hold negative numbers, such as `np.int16`.)
# 3. Make `amount_of_change`: how much each pixel changed, whatever the
#    direction, `[2, 10, 0]`.

# %% tags=["solution"]
counts_half = counts.astype(float) + 0.5  # or counts + 0.5, which makes a new float array
change = frame_b.astype(np.int16) - frame_a.astype(np.int16)
amount_of_change = np.abs(change)
print(counts_half)
print(change, change.dtype, amount_of_change)

# %% [markdown]
# ## 2. Aggregations and broadcasting
#
# ### 2a. Aggregations along more than one axis
#
# `spikes` holds spike counts from a (simulated) experiment: 4 neurons, each
# recorded in 5 trials, with each trial split into 10 time bins.
# `spikes[n, t, b]` is the number of spikes neuron `n` fired in bin `b` of
# trial `t`.

# %%
spikes = rng.poisson(3, size=(4, 5, 10))
spikes.shape

# %% [markdown]
# Make:
#
# 1. `total_per_neuron`: the total number of spikes from each neuron, over all
#    trials and bins (4 values). (Hint: `axis` also accepts a tuple of axes.)
# 2. `psth`: for each neuron, the mean count in each time bin, averaged over
#    trials. Shape (4, 10).
# 3. `busiest_trial`: for each neuron, the **number** of the trial in which it
#    fired the most spikes (4 values).
# 4. `overall_mean`: the mean count over the whole array (one number)

# %% tags=["solution"]
total_per_neuron = spikes.sum(axis=(1, 2))
psth = spikes.mean(axis=1)
busiest_trial = spikes.sum(axis=2).argmax(axis=1)
overall_mean = spikes.mean()
print(total_per_neuron, psth.shape, busiest_trial, overall_mean, sep="\n")

# %% [markdown]
# ### 2b. Broadcasting
#
# No loops in this section.
#
# 1. `points` holds the (x, y) positions of 5 cells. Make `distances`, the
#    5 x 5 array where `distances[i, j]` is the distance between cell `i` and
#    cell `j`. (Hint: give one copy of `points` the shape (5, 1, 2) and another
#    the shape (1, 5, 2). Their difference has shape (5, 5, 2). Then square, sum
#    over the last axis and take the square root.) Check that the diagonal is 0.

# %%
points = rng.random((5, 2))

# %% tags=["solution"]
offsets = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt((offsets**2).sum(axis=-1))
print(distances.round(2))

# %% [markdown]
# 2. `traces` holds the fluorescence of 3 neurons over 50 time points, with a
#    response from time point 30. Make `dff`, the change relative to each
#    neuron's baseline, `(traces - baseline) / baseline`, where `baseline` is
#    the mean of that neuron's **first 10** time points. `dff[:, 30:35]` should
#    be about 0.4.

# %%
traces = rng.normal(100, 5, size=(3, 50))
traces[:, 30:35] += 40

# %% tags=["solution"]
baseline = traces[:, :10].mean(axis=1, keepdims=True)  # shape (3, 1)
dff = (traces - baseline) / baseline
print(dff[:, 28:36].round(2))

# %% [markdown]
# 3. **(SPL Exercise 22, part 2)** Divide each **column** of `a` element-wise
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
# 4. The cell below should weight each **trial** of `spikes` (shape
#    (4, 5, 10)) by `trial_weights`, but it fails. Fix it.

# %% tags=["raises-exception"]
trial_weights = np.array([1.0, 1.0, 0.5, 0.5, 0.0])
spikes * trial_weights

# %% tags=["solution"]
# Shapes are compared from the right: (4, 5, 10) against (5,) lines up 10
# against 5. Making the weights (5, 1) lines 5 up with the trial axis instead.
weighted = spikes * trial_weights[:, np.newaxis]
weighted.shape

# %% [markdown]
# ## 3. Masks, fancy indexing and vectorising
#
# ### 3a. Boolean masks
#
# `rt` holds reaction times (in ms) from 40 trials of a task, and `correct`
# says whether the answer on each trial was right. A few trials are odd.

# %%
rt = rng.normal(450, 120, size=40).round()
rt[[5, 21, 33]] = [95, 120, 1450]
correct = rng.random(40) < 0.8
rt[:10], correct[:10]

# %% [markdown]
# Make:
#
# 1. `too_fast`: the reaction times below 150 ms (anticipations)
# 2. `n_correct`: the number of correct trials
# 3. `mean_rt_correct`: the mean reaction time of the **correct** trials only
# 4. `slow_errors`: the reaction times of trials that were **wrong** and slower
#    than 500 ms
# 5. `rt_clean`: a **new** array in which every reaction time below 150 or above
#    1000 is replaced by `np.nan`. `rt` must not change. Then compare
#    `rt.mean()` with `np.nanmean(rt_clean)`.

# %% tags=["solution"]
too_fast = rt[rt < 150]
n_correct = correct.sum()
mean_rt_correct = rt[correct].mean()
slow_errors = rt[~correct & (rt > 500)]
rt_clean = np.where((rt < 150) | (rt > 1000), np.nan, rt)
print(too_fast, n_correct, mean_rt_correct, slow_errors, sep="\n")
print(rt.mean(), np.nanmean(rt_clean))

# %% [markdown]
# 6. Using `spikes` from part 2, make `full_trials`: for neuron 0, the trials
#    (whole rows of `spikes[0]`) in which it fired in **every** time bin.
#    (Hint: `np.all` with an `axis`.)

# %% tags=["solution"]
full_trials = spikes[0][np.all(spikes[0] > 0, axis=1)]
print(full_trials)

# %% [markdown]
# ### 3b. Fancy indexing
#
# An integer array can index into a **lookup table**: every code is replaced
# by the matching entry.
#
# 1. Each trial of an experiment has a condition code, 0, 1 or 2. Make
#    `condition_names`: the name of each trial's condition, from `conditions`.

# %%
conditions = np.array(["rest", "left", "right"])
codes = rng.integers(0, 3, size=12)
codes

# %% tags=["solution"]
condition_names = conditions[codes]
print(condition_names)

# %% [markdown]
# 2. Make `rt_sorted`, the reaction times from 3a sorted fastest first, and
#    `correct_sorted`, the `correct` values reordered **the same way** so that they
#    still belong to the same trials. (Hint: `argsort`.)

# %% tags=["solution"]
order = rt.argsort()
rt_sorted = rt[order]
correct_sorted = correct[order]
print(rt_sorted[:5], correct_sorted[:5])

# %% [markdown]
# 3. `labels` is a small segmented image: 0 is background, 1 and 2 are two
#    cells. `palette` gives one RGB colour per label. Make `colour_image`, an
#    RGB version of `labels`, using **one** indexing expression. What shape is
#    it, and why?
#
# Then, in place:
#
# 4. Set every pixel in the first and last **rows** of `labels` to 0, using one
#    integer list as the index.

# %%
labels = np.array(
    [
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
    ]
)
palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]])  # black, red, blue

# %% tags=["solution"]
colour_image = palette[labels]
print(colour_image.shape)
print(colour_image[1])

# %% [markdown] tags=["answer"]
# `(4, 5, 3)`: every entry of the (4, 5) `labels` array is replaced by a row of
# `palette`, which has 3 values. Fancy indexing gives the shape of the index
# array, followed by the remaining axes of the array being indexed.

# %% tags=["solution"]
labels[[0, -1]] = 0
print(labels)

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
# ### 3c. Vectorised versus loops
#
# `path` holds the (x, y) position of an animal at 100 000 time points (a random
# walk). Its **path length** is the sum of the distances between consecutive
# positions.
#
# Write `path_length_loop(path)` using a `for` loop over the positions, and
# `path_length_numpy(path)` using no loop at all. Check that they agree, then
# time both with `%timeit`. How many times faster is NumPy? (Hint for the NumPy
# version: `path[1:] - path[:-1]` gives every step at once. See also the
# Handbook's [Profiling and Timing
# Code](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html).)

# %%
path = rng.normal(0, 1, size=(100_000, 2)).cumsum(axis=0)


def path_length_loop(path):
    # BEGIN SOLUTION
    total = 0.0
    for k in range(1, len(path)):
        dx = path[k, 0] - path[k - 1, 0]
        dy = path[k, 1] - path[k - 1, 1]
        total += (dx**2 + dy**2) ** 0.5
    return total
    # END SOLUTION


def path_length_numpy(path):
    # BEGIN SOLUTION
    steps = path[1:] - path[:-1]
    return np.sqrt((steps**2).sum(axis=1)).sum()
    # END SOLUTION


# %% tags=["solution"]
np.isclose(path_length_loop(path), path_length_numpy(path))

# %% tags=["solution"]
# %timeit -n 3 -r 3 path_length_loop(path)
# %timeit -n 3 -r 3 path_length_numpy(path)

# %% [markdown] tags=["answer"]
# On a typical laptop the loop takes tens of milliseconds and NumPy under one:
# roughly 50 times faster. The loop pays for Python work at every time point
# (indexing, arithmetic, `** 0.5`); NumPy does each step once, over the whole
# array, in compiled code.

# %% [markdown]
# ### 3d. Data statistics (SPL Exercise 23, parts 1 to 5)
#
# `populations.txt` describes the populations of hares and lynxes (and carrots)
# in northern Canada over 21 years. The first column is the year.

# %%
data = np.loadtxt("../data/populations.txt")
year, hares, lynxes, carrots = data.T  # trick: columns to variables
populations = data[:, 1:]
species = np.array(["Hare", "Lynx", "Carrot"])
data[:5]

# %% [markdown]
# Compute and print, **without any for-loops**:
#
# 1. The mean and standard deviation of the population of each species over the
#    whole period.
# 2. The year in which each species had its largest population.
# 3. Which species has the largest population in each year. (Hint: `argmax` and
#    fancy indexing of `species`.)
# 4. The years in which any of the populations is above 50000. (Hint:
#    comparisons and `np.any`.)
# 5. The two years in which each species had its lowest populations. (Hint:
#    `argsort` and fancy indexing.)
#
# (Part 6 needs a plot, so it comes on Day 2.)

# %% tags=["solution"]
# 1.
print("            Hares     Lynxes    Carrots")
print("Mean:", populations.mean(axis=0))
print("Std: ", populations.std(axis=0))

# %% tags=["solution"]
# 2.
print("Year of maximum:", year[populations.argmax(axis=0)])

# %% tags=["solution"]
# 3.
print(np.column_stack([year.astype(int), species[populations.argmax(axis=1)]]))

# %% tags=["solution"]
# 4.
print("Any above 50000:", year[np.any(populations > 50000, axis=1)])

# %% tags=["solution"]
# 5. argsort sorts each column; the first two rows hold the indices of the two lowest values.
print("Two lowest years per species (columns: hare, lynx, carrot):")
print(year[populations.argsort(axis=0)[:2]])

# %% [markdown]
# ## 4. Stretch: crude integral approximations (SPL Exercise 24)
#
# Write a function `f(a, b, c)` that returns $a^b - c$. Form a 24 x 12 x 6 array
# containing its values over the parameter ranges `[0, 1] x [0, 1] x [0, 1]`.
#
# Approximate the 3-D integral
#
# $$\int_0^1\int_0^1\int_0^1 (a^b - c)\, da\, db\, dc$$
#
# by the **mean** of that array. The exact result is $\ln 2 - \frac{1}{2} \approx
# 0.1931$. What is your relative error?
#
# Hints: use element-wise operations and broadcasting. `np.ogrid[0:1:20j]` gives 20
# points in the range [0, 1].

# %%
def f(a, b, c):
    # BEGIN SOLUTION
    return a**b - c
    # END SOLUTION


# %% tags=["solution"]
a = np.linspace(0, 1, 24)
b = np.linspace(0, 1, 12)
c = np.linspace(0, 1, 6)
samples = f(a[:, np.newaxis, np.newaxis], b[np.newaxis, :, np.newaxis], c[np.newaxis, np.newaxis, :])

# Equivalent, using ogrid:
# a, b, c = np.ogrid[0:1:24j, 0:1:12j, 0:1:6j]
# samples = f(a, b, c)

approx = samples.mean()
exact = np.log(2) - 0.5
print(samples.shape)
print(f"Approximation: {approx:.4f}, exact: {exact:.4f}, relative error: {abs(approx - exact) / exact:.1%}")

# %% [markdown] tags=["solution"]
# The relative error is about 2%. With more points (try 240 x 120 x 60) it shrinks,
# at the cost of 1000 times more memory: broadcasting builds the whole grid.

# %% [markdown]
# ## 5. Stretch: Markov chain (SPL Exercise 26)
#
# A Markov chain has a transition matrix `P` and a probability distribution `p`
# over its states:
#
# 1. `0 <= P[i, j] <= 1` is the probability of going from state `i` to state `j`.
# 2. Transition rule: $p_{new} = P^T p_{old}$.
# 3. Normalisation: every row of `P` sums to 1, and `p.sum() == 1`.
#
# For 5 states:
#
# * Construct a random matrix and normalise each row so it is a transition matrix.
# * Start from a random (normalised) probability distribution `p` and take 50
#   steps, giving `p_50`.
# * Compute the stationary distribution: the eigenvector of `P.T` with eigenvalue
#   1 (numerically, the one closest to 1), giving `p_stationary`. Remember to
#   normalise it.
# * Check whether `p_50` and `p_stationary` are equal to within a tolerance of 1e-5.
#
# Toolbox: `rng.random`, `@`, `np.linalg.eig`, reductions, `abs`, `argmin`,
# comparisons, `np.all`, `np.linalg.norm`. (One `for` loop for the 50 steps is fine.)

# %% tags=["solution"]
n_states = 5
P = rng.random((n_states, n_states))
P /= P.sum(axis=1)[:, np.newaxis]  # normalise rows

p = rng.random(n_states)
p /= p.sum()

for _ in range(50):
    p = P.T @ p
p_50 = p

w, v = np.linalg.eig(P.T)
j_stationary = np.argmin(abs(w - 1.0))
p_stationary = v[:, j_stationary].real
p_stationary /= p_stationary.sum()

print(p_50)
print(p_stationary)
print("Equal to 1e-5:", np.all(abs(p_50 - p_stationary) < 1e-5))
