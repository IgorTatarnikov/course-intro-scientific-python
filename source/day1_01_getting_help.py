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
# Look up `np.linspace` three ways:
#
# 1. `help(np.linspace)`, which works in any Python session
# 2. `np.linspace?`, which works in Jupyter and IPython (and `np.linspace??` shows the source)
# 3. The online docs: search for "numpy linspace", or go to <https://numpy.org/doc/stable/>
#
# Then answer the questions below.

# %%
import numpy as np

# %%
help(np.linspace)

# %%
# np.linspace?

# %% [markdown]
# **Questions**
#
# 1. How many points does `np.linspace(0, 1)` return if you do not pass `num`?
# 2. What does `endpoint=False` change?
# 3. Which argument makes `linspace` also return the spacing between points?

# %% [markdown] tags=["answer"]
# 1. 50: the default is `num=50`.
# 2. The stop value is left out, so the points are spaced by `(stop - start) / num`
#    instead of `(stop - start) / (num - 1)`.
# 3. `retstep=True` returns a tuple `(samples, step)`.

# %% tags=["solution"]
print(len(np.linspace(0, 1)))
print(np.linspace(0, 1, 5))
print(np.linspace(0, 1, 5, endpoint=False))
print(np.linspace(0, 1, 5, retstep=True))

# %% [markdown]
# ## 3. Read a traceback
#
# The cell below fails on purpose. Read the error from the **bottom up**:
#
# 1. What type of error is it?
# 2. Which argument caused it, and why?
# 3. Fix the call in the empty cell underneath.

# %% tags=["raises-exception"]
n_points = 100 / 8
x = np.linspace(0, 1, n_points)

# %% [markdown] tags=["answer"]
# 1. A `TypeError`.
# 2. `num` must be an integer, but `100 / 8` is the float `12.5`. Python's `/`
#    always returns a float; use `//` or `int()` for a whole number.

# %% tags=["solution"]
n_points = 100 // 8
x = np.linspace(0, 1, n_points)
x.shape

# %% [markdown]
# ## 4. Tab completion and searching
#
# * Type `np.lin` in a cell and press <kbd>Tab</kbd>. What other functions start with `lin`?
# * `np.lookfor` was removed in NumPy 2, so search for a topic in the online docs instead.
#   Search for "inverse of a matrix": which function do you find?

# %% [markdown] tags=["answer"]
# * `np.linspace` and `np.linalg` (the linear algebra submodule).
# * `np.linalg.inv`. (`np.linalg.pinv` is the pseudo-inverse, for matrices that have no inverse.)
