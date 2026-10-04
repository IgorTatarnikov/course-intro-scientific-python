# %% [markdown]
# # Day 1 · Setup check and finding help
#
# **Block:** Welcome, setup check and help (10 min)
#
# How the exercises work: try each one yourself first. Cells that say
# `# Your code here` or *Your answer here* are yours to fill in. The solutions are
# released after each block.
#
# Material adapted from [Scientific Python Lectures, Getting help and finding
# documentation](https://lectures.scientific-python.org/intro/help/help.html)
# (CC BY 4.0).

# %% [markdown]
# ## 1. Check your setup
#
# Run the cell below. It runs the same check script you ran before the course.
# Every line should say `OK`; if not, put up your hand.

# %%
# %run ../check_setup.py

# %% [markdown]
# ## 2. Three ways to read the documentation
#
# On the slides we looked up `np.linspace`. Now look up **`np.arange`** the same
# three ways:
#
# 1. `help(np.arange)`, which works in any Python session
# 2. `np.arange?`, which works in Jupyter and IPython (and `np.arange??` tries to show the source)
# 3. The online docs: search for "numpy arange", or go to <https://numpy.org/doc/stable/>
#
# Then answer the questions below.

# %%
import numpy as np

# %%
help(np.arange)

# %%
# np.arange?

# %% [markdown]
# **Questions**
#
# 1. Is the `stop` value included in the result? How is that different from `np.linspace`?
# 2. What does `np.arange(3, 10, 2)` return? Predict first, then run it.
# 3. What does the documentation recommend using instead when the step is not a
#    whole number, such as 0.1?
# 4. Which argument sets the dtype of the result?

# %% [markdown] tags=["answer"]
# 1. No: `arange` stops *before* `stop`, like `range()`. `np.linspace` includes the
#    stop value unless you pass `endpoint=False`.
# 2. `[3, 5, 7, 9]`.
# 3. `np.linspace`: with a float step, rounding can change how many values you get.
# 4. `dtype`, for example `np.arange(5, dtype=float)`.

# %% tags=["solution"]
print(np.arange(0, 1, 0.25))
print(np.arange(3, 10, 2))
print(np.arange(5, dtype=float))

# %% [markdown]
# ## 3. Read a traceback
#
# The cell below fails on purpose. The error happens inside a function, which is
# called from another function. Read the traceback from the **bottom up**:
#
# 1. What type of error is it, and what does the message say?
# 2. In which function, and on which line, did it break?
# 3. Which line in the cell started the chain of calls?
# 4. Fix it in the empty cell underneath, **without changing either function**.

# %% tags=["raises-exception"]
def peak_to_peak(values):
    return values.max() - values.min()


def report(name, values):
    print(name, "range:", peak_to_peak(values))


report("weights", [61.2, 74.5, 58.9])

# %% [markdown] tags=["answer"]
# 1. An `AttributeError`: `'list' object has no attribute 'max'`.
# 2. In `peak_to_peak`, on the line `return values.max() - values.min()`.
# 3. The last line, `report("weights", ...)`, which calls `report`, which calls `peak_to_peak`.
# 4. A Python list has no `.max()` method; a NumPy array does. Pass an array instead.

# %% tags=["solution"]
report("weights", np.array([61.2, 74.5, 58.9]))

# %% [markdown]
# ## 4. Tab completion and searching
#
# * Type `np.cum` in a cell and press <kbd>Tab</kbd>. Which functions start with `cum`?
#   Pick one and use `?` to find out what it does.
# * `np.lookfor` was removed in NumPy 2, so search for a topic in the online docs instead.
#   Search for "inverse of a matrix": which function do you find?

# %% [markdown] tags=["answer"]
# * `np.cumsum` and `np.cumprod` (running sum and product), and their newer names
#   `np.cumulative_sum` and `np.cumulative_prod`.
# * `np.linalg.inv`. (`np.linalg.pinv` is the pseudo-inverse, for matrices that have no inverse.)
