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
# ### 1a. Fetching parts of an array
#
# All the tasks in this section use `grid`:

# %%
grid = np.arange(1, 37).reshape(6, 6)
grid

# %% [markdown]
# Using **one indexing expression each** (no loops, no typing values in),
# make:
#
# 1. `value`: the number in row 2, column 3 (counting from 0). It should be 16.
# 2. `last_row`: the whole last row
# 3. `third_column`: the whole third column
# 4. `every_other`: every other column, starting with the first
# 5. `centre`: the central 2 x 2 block, `[[15, 16], [21, 22]]`
# 6. `bottom_right`: the bottom-right 3 x 3 block
# 7. `upside_down`: `grid` with its rows in reverse order
# 8. `corners`: the four corners, as a 2 x 2 array `[[1, 6], [31, 36]]`
#    (Hint: a step can be larger than 1.)

# %%
# BEGIN SOLUTION
value = grid[2, 3]
last_row = grid[-1]
third_column = grid[:, 2]
every_other = grid[:, ::2]
centre = grid[2:4, 2:4]
bottom_right = grid[-3:, -3:]
upside_down = grid[::-1]
corners = grid[::5, ::5]
# END SOLUTION

# %% tags=["solution"]
print(value, last_row, third_column, sep="\n")
print(every_other, centre, bottom_right, upside_down, corners, sep="\n\n")

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
# 1. Make `top_left`, the top-left 2 x 2 block of `grid`, and set all its
#    values to `-1`. Print `grid`: did it change?
# 2. Reset `grid`. Now do the same again, but this time **without changing
#    `grid`**.

# %%
grid = np.arange(1, 37).reshape(6, 6)

# %% tags=["solution"]
# 1. A slice is a view, so grid changes too.
top_left = grid[:2, :2]
top_left[:] = -1
print(grid)

# %% tags=["solution"]
# 2. Copy first.
grid = np.arange(1, 37).reshape(6, 6)
top_left = grid[:2, :2].copy()
top_left[:] = -1
print(grid)

# %% [markdown]
# 3. Which of these share memory with `grid`, and which are copies? Find out
#    with `np.shares_memory(grid, ...)` rather than guessing:
#    `grid[1:3]`, `grid[:, 1]`, `grid[[1, 2]]`, `grid[grid > 10]`, `grid.T`,
#    `grid.reshape(4, 9)`.

# %% tags=["solution"]
for name, candidate in [
    ("grid[1:3]", grid[1:3]),
    ("grid[:, 1]", grid[:, 1]),
    ("grid[[1, 2]]", grid[[1, 2]]),
    ("grid[grid > 10]", grid[grid > 10]),
    ("grid.T", grid.T),
    ("grid.reshape(4, 9)", grid.reshape(4, 9)),
]:
    print(f"{name:20} shares memory: {np.shares_memory(grid, candidate)}")

# %% [markdown] tags=["answer"]
# Slices, `.T` and `reshape` are views. Indexing with a list of integers
# (fancy indexing) or with a mask always makes a copy.

# %% [markdown]
# ### 1d. In place, or a new array?
#
# Each task below starts from a fresh `grid`. Use **one line** for each.
#
# 1. Set the whole last column of `grid` to 0, in place.
# 2. Add 100 to every value in row 1, in place.
# 3. Make `doubled`, a new array with every value of `grid` doubled. `grid`
#    must not change.

# %%
grid = np.arange(1, 37).reshape(6, 6)

# %% tags=["solution"]
grid[:, -1] = 0
grid[1] += 100
doubled = grid * 2
print(grid)
print(doubled)

# %% [markdown]
# `x` and `y` below are two names for the **same** array.
#
# 4. Double `x` so that `y` sees the change too. Print both.
# 5. Reset them, then double `x` so that `y` keeps the old values.

# %%
x = np.arange(5)
y = x

# %% tags=["solution"]
# 4. In place: x and y still point at the same array.
x *= 2
print(x, y)

# %% tags=["solution"]
# 5. x * 2 makes a new array, and the name x now points at it.
x = np.arange(5)
y = x
x = x * 2
print(x, y)

# %% [markdown]
# 6. Make `sorted_values`, a sorted copy of `values`, leaving `values`
#    unchanged. Print `values` to check.
# 7. Now sort `values` itself, in place. What does the method return?

# %%
values = rng.integers(0, 100, size=8)
values

# %% tags=["solution"]
sorted_values = np.sort(values)
print(values, sorted_values)

# %% tags=["solution"]
result = values.sort()
print(values, result)

# %% [markdown] tags=["answer"]
# `np.sort(values)` returns a new array. The method `values.sort()` sorts in
# place and returns `None`, so `values = values.sort()` would throw your data
# away.

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
# `pixels` holds three pixel brightnesses stored as `uint8` (0 to 255), as
# images usually are.
#
# 2. Make `brighter`: every pixel 100 brighter, but **capped at 255** instead
#    of wrapping round. `brighter` should still be `uint8`, and `pixels` must
#    not change. (Hint: `.astype()` and `np.clip`.)

# %%
pixels = np.array([10, 200, 250], dtype=np.uint8)
pixels + 100  # wraps round: not what we want

# %% tags=["solution"]
counts_half = counts.astype(float) + 0.5  # or counts + 0.5, which makes a new float array
brighter = np.clip(pixels.astype(np.int16) + 100, 0, 255).astype(np.uint8)
print(counts_half)
print(brighter, brighter.dtype, pixels)

# %% [markdown]
# ## 2. Aggregations and broadcasting
#
# ### 2a. Aggregations
#
# `scores` holds the marks of 6 students (rows) in 4 tests (columns).
# Make:
#
# 1. `student_mean`: the mean mark of each student (6 values)
# 2. `test_best`: the best mark in each test (4 values)
# 3. `test_winner`: the **row number** of the student who did best in each test
# 4. `overall_mean`: the mean of every mark in the table (one number)

# %%
scores = rng.integers(40, 101, size=(6, 4))
scores

# %% tags=["solution"]
student_mean = scores.mean(axis=1)
test_best = scores.max(axis=0)
test_winner = scores.argmax(axis=0)
overall_mean = scores.mean()
print(student_mean, test_best, test_winner, overall_mean, sep="\n")

# %% [markdown]
# ### 2b. Broadcasting
#
# No loops in this section.
#
# 1. `times_table`: the 10 x 10 multiplication table, where `times_table[i, j]`
#    is `(i + 1) * (j + 1)`. (Hint: one row and one column from `np.arange`.)
# 2. `scores_centred`: `scores` with each **test's** mean subtracted, so every
#    column has mean 0.
# 3. `scores_fraction`: each student's marks divided by that student's total,
#    so every **row** sums to 1. (Hint: `keepdims=True`.)

# %% tags=["solution"]
n = np.arange(1, 11)
times_table = n[:, np.newaxis] * n
scores_centred = scores - scores.mean(axis=0)
scores_fraction = scores / scores.sum(axis=1, keepdims=True)
print(times_table)
print(scores_centred.mean(axis=0).round(10))
print(scores_fraction.sum(axis=1))

# %% [markdown]
# 4. **(SPL Exercise 22, part 2)** Divide each **column** of `a` element-wise
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
# 5. The cell below fails. Fix it so that 0 is added to the first row of
#    `ones`, 1 to the second and 2 to the third.

# %% tags=["raises-exception"]
ones = np.ones((3, 2))
ones + np.arange(3)

# %% tags=["solution"]
# Broadcasting compares shapes from the right: (3, 2) against (3,) lines up
# 2 against 3. Making the second operand (3, 1) lines 3 up with 3 instead.
ones + np.arange(3)[:, np.newaxis]

# %% [markdown]
# ## 3. Masks, fancy indexing and vectorising
#
# ### 3a. Boolean masks
#
# `temps` holds 30 daily temperatures.

# %%
temps = rng.normal(8, 7, size=30).round(1)
temps

# %% [markdown]
# Make:
#
# 1. `warm`: only the temperatures above 15
# 2. `n_frost`: the number of days below 0
# 3. `mild`: the temperatures from 5 to 15, inclusive
# 4. `no_frost`: a **new** array in which every negative temperature is
#    replaced by 0. `temps` must not change. (Hint: `np.where`, or a copy.)
#
# Then:
#
# 5. Replace every negative value **in `temps` itself** by 0, in one line.

# %% tags=["solution"]
warm = temps[temps > 15]
n_frost = (temps < 0).sum()
mild = temps[(temps >= 5) & (temps <= 15)]
no_frost = np.where(temps < 0, 0, temps)
print(warm, n_frost, mild, no_frost, temps, sep="\n")

# %% tags=["solution"]
temps[temps < 0] = 0
print(temps)

# %% [markdown]
# 6. Using `scores` from part 2, make `top_students`: the rows (whole rows)
#    of students who scored **above 90 in at least one test**. (Hint:
#    `np.any` with an `axis`.)

# %% tags=["solution"]
top_students = scores[np.any(scores > 90, axis=1)]
print(top_students)

# %% [markdown]
# ### 3b. Fancy indexing

# %%
grid = np.arange(1, 37).reshape(6, 6)
grid

# %% [markdown]
# Make:
#
# 1. `some_rows`: rows 0, 2 and 5 of `grid`
# 2. `shuffled`: `grid` with its columns in the order 3, 0, 5, 1, 4, 2
# 3. `anti_diagonal`: the diagonal from the top-right corner to the
#    bottom-left, `[6, 11, 16, 21, 26, 31]`. (Hint: one array of row numbers
#    and one of column numbers.)
#
# Then, in place:
#
# 4. Set the main diagonal of `grid` (top-left to bottom-right) to 0.

# %% tags=["solution"]
some_rows = grid[[0, 2, 5]]
shuffled = grid[:, [3, 0, 5, 1, 4, 2]]
rows = np.arange(6)
anti_diagonal = grid[rows, rows[::-1]]
print(some_rows, shuffled, anti_diagonal, sep="\n\n")

# %% tags=["solution"]
grid[rows, rows] = 0  # or np.fill_diagonal(grid, 0)
print(grid)

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
# Write `sum_of_squares_loop(values)` using a `for` loop, and
# `sum_of_squares_numpy(values)` using no loop at all. Check that they agree, then
# time both with `%timeit` on one million values. How many times faster is NumPy?
# (See the Handbook's [Profiling and Timing
# Code](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html).)

# %%
values = rng.random(1_000_000)


def sum_of_squares_loop(values):
    # BEGIN SOLUTION
    total = 0.0
    for v in values:
        total += v * v
    return total
    # END SOLUTION


def sum_of_squares_numpy(values):
    # BEGIN SOLUTION
    return np.sum(values**2)
    # END SOLUTION


# %% tags=["solution"]
np.isclose(sum_of_squares_loop(values), sum_of_squares_numpy(values))

# %% tags=["solution"]
# %timeit -n 3 -r 3 sum_of_squares_loop(values)
# %timeit -n 3 -r 3 sum_of_squares_numpy(values)

# %% [markdown] tags=["answer"]
# On a typical laptop the loop takes tens of milliseconds and NumPy well under
# one: roughly 50 to 100 times faster. `values @ values` (a dot product) is
# faster still, because it never allocates the `values**2` temporary array.

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
