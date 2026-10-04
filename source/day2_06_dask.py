# %% [markdown]
# # Day 2 · Whirlwind tour: dask
#
# **Block:** whirlwind tour, dask (about 10 min: demo, then try it)
#
# **Problem it solves:** arrays (and DataFrames) **bigger than memory**, and using
# all your CPU cores. A dask array is a grid of NumPy **chunks**. Operations build
# a **task graph** lazily, and nothing runs until you call `.compute()`.
#
# Material follows the [dask tutorial](https://tutorial.dask.org) and the
# [dask.array docs](https://docs.dask.org/en/stable/array.html).

# %%
import time

import dask.array as da
import numpy as np

# %% [markdown]
# ## Demo
#
# A 20,000 x 20,000 array of float64 would take 3.2 GB as a NumPy array. As a dask
# array it costs nothing until we compute, and then it is processed chunk by chunk:

# %%
x = da.random.random((20_000, 20_000), chunks=(2_000, 2_000))
x

# %% [markdown]
# Operations look exactly like NumPy, but return instantly, because they only
# add tasks to the graph:

# %%
y = (x - x.mean(axis=0)) ** 2
result = y.mean()
result

# %%
print("tasks in the graph:", len(result.__dask_graph__()))

# %% [markdown]
# `.compute()` runs the graph in parallel, holding only a few chunks in memory at
# a time:

# %%
start = time.perf_counter()
value = result.compute()
print(f"{value:.5f} in {time.perf_counter() - start:.1f} s")

# %% [markdown]
# For uniform random numbers, the variance is 1/12 ≈ 0.08333, which is a handy
# check.
#
# **dask behind xarray.** Pass `chunks=` when opening a file, and every xarray
# operation becomes lazy and chunked:

# %%
import xarray as xr

ds = xr.open_dataset("../data/air_temperature_daily.nc", chunks={"time": 100})
ds["air"]

# %%
ds["air"].mean(dim="time").compute().shape

# %% [markdown]
# ## Try it
#
# 1. Make a dask array of shape (20,000, 5,000) with chunks of (2,000, 5,000).
#    How big would it be in NumPy? (`x.nbytes / 1e9` gives gigabytes.)
# 2. Compute the standard deviation of each column. Time it.
# 3. Repeat with much smaller chunks, (250, 250), and much larger ones, (20,000,
#    5,000). What happens to the number of tasks and to the run time? Why?

# %% tags=["solution"]
for chunks in [(2_000, 5_000), (250, 250), (20_000, 5_000)]:
    a = da.random.random((20_000, 5_000), chunks=chunks)
    std = a.std(axis=0)
    start = time.perf_counter()
    std.compute()
    elapsed = time.perf_counter() - start
    print(
        f"chunks {str(chunks):>16}: {a.nbytes / 1e9:.1f} GB, "
        f"{len(std.__dask_graph__()):5d} tasks, {elapsed:.2f} s"
    )

# %% [markdown] tags=["answer"]
# The array is 0.8 GB as NumPy. On an 8-core laptop we measured:
#
# | chunks | chunk size | tasks | time | peak memory |
# |---|---|---|---|---|
# | (2,000, 5,000) | 80 MB | 25 | 0.4 s | 1.8 GB |
# | (250, 250) | 0.5 MB | 3,780 | 0.7 s | 0.2 GB |
# | (20,000, 5,000) | 800 MB | 4 | 0.7 s | 2.6 GB |
#
# Your numbers will differ, but the shape of the result usually holds. Huge
# chunks give too few tasks to keep every core busy, and each task needs a lot of
# memory: one 800 MB chunk plus its temporaries. Tiny chunks use very little
# memory, but give so many tasks that the scheduler's fixed cost per task starts to
# dominate. In between there is a broad sweet spot. The [dask
# docs](https://docs.dask.org/en/stable/array-chunks.html) suggest starting
# around 100 MB per chunk, with at least a few chunks per core.
#
# **Where next:** the [dask tutorial](https://tutorial.dask.org), `dask.dataframe`
# for pandas-like tables too big for memory, and the distributed scheduler's
# live dashboard (`from dask.distributed import Client; client = Client()`),
# which shows every task as it runs.
