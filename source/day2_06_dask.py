# %% [markdown]
# # Day 2 · Whirlwind tour: dask
#
# **Block:** whirlwind tour, dask (about 10 min: demo, then try it)
#
# **Problem it solves:** arrays (and DataFrames) **bigger than memory**, and using
# all your CPU cores. A dask array is a grid of NumPy **chunks**. Operations build
# a **task graph** lazily, and nothing runs until you call `.compute()`.
#
# Our 10 s slice of LFP is 4 MB. The session it comes from is not: its LFP
# file on DANDI holds 67 minutes of all 384 channels at 2500 Hz, and the raw
# spike band (30 000 samples per second) makes the whole file 51 GB. That does
# not fit in a laptop's memory, but it can still be processed chunk by chunk.
#
# Material follows the [dask tutorial](https://tutorial.dask.org) and the
# [dask.array docs](https://docs.dask.org/en/stable/array.html).

# %%
import time

import dask.array as da
import h5py
import numpy as np

# %% [markdown]
# ## Demo
#
# A stand-in for 10 minutes of LFP at the original 2500 Hz, as float64: 1.5
# million samples by 384 channels would take 4.6 GB as a NumPy array. As a dask
# array it costs nothing until we compute, and then it is processed chunk by
# chunk:

# %%
x = da.random.normal(0, 100, size=(1_500_000, 384), chunks=(25_000, 384))
x

# %% [markdown]
# Operations look exactly like NumPy, but return instantly, because they only
# add tasks to the graph. Here: subtract a common reference (the mean over
# channels at each time point; on Day 1 we used the median), then take each
# channel's standard deviation:

# %%
referenced = x - x.mean(axis=1, keepdims=True)
channel_std = referenced.std(axis=0)
channel_std

# %%
print("tasks in the graph:", len(channel_std.__dask_graph__()))

# %% [markdown]
# `.compute()` runs the graph in parallel, holding only a few chunks in memory at
# a time:

# %%
start = time.perf_counter()
values = channel_std.compute()
print(f"{values.shape}, mean {values.mean():.1f} in {time.perf_counter() - start:.1f} s")

# %% [markdown]
# The values are just under 100, the standard deviation we asked for: subtracting
# the mean over 384 channels removes a little of each one's own noise, which is
# a handy check.
#
# **Real files, read lazily.** `da.from_array` wraps an HDF5 dataset without
# reading it. Each task reads its own chunk from disk, so this works the same
# on our 4 MB file or on the full 51 GB one. The file has to stay open until the
# computation is done:

# %%
with h5py.File("../data/ibl_session.h5") as f:
    lazy = da.from_array(f["lfp/data"], chunks=(1000, 384))
    print(lazy)
    std_uv = (lazy * f["lfp"].attrs["uV_per_count"]).std(axis=0).compute()
std_uv[:5].round(1)

# %% [markdown]
# **dask behind xarray.** Wrap a dask array in a `DataArray` (or call `.chunk()`
# on one, or pass `chunks=` to `xr.open_dataset`), and every xarray operation
# becomes lazy and chunked.
#
# ## Try it
#
# 1. Make a dask array standing in for 80 s of LFP at 2500 Hz: shape
#    (200 000, 384), with chunks of (25 000, 384). How big would it be in
#    NumPy? (`x.nbytes / 1e9` gives gigabytes.)
# 2. Compute the standard deviation of each channel. Time it.
# 3. Repeat with much smaller chunks, (500, 48), and one chunk for the whole
#    array, (200 000, 384). What happens to the number of tasks and to the run
#    time? Why?

# %% tags=["solution"]
for chunks in [(25_000, 384), (500, 48), (200_000, 384)]:
    a = da.random.normal(0, 100, size=(200_000, 384), chunks=chunks)
    std = a.std(axis=0)
    start = time.perf_counter()
    std.compute()
    elapsed = time.perf_counter() - start
    print(
        f"chunks {str(chunks):>14}: {a.nbytes / 1e9:.2f} GB, "
        f"{len(std.__dask_graph__()):5d} tasks, {elapsed:.2f} s"
    )

# %% [markdown] tags=["answer"]
# The array is 0.6 GB as NumPy. On an 8-core laptop we measured:
#
# | chunks | chunk size | tasks | time | peak memory |
# |---|---|---|---|---|
# | (25 000, 384) | 77 MB | 20 | 0.5 s | 1.3 GB |
# | (500, 48) | 0.2 MB | 7 488 | 1.2 s | 0.2 GB |
# | (200 000, 384) | 614 MB | 4 | 1.5 s | 2.0 GB |
#
# Your numbers will differ, but the shape of the result usually holds. One huge
# chunk gives too few tasks to keep every core busy, and needs a lot of memory:
# the whole chunk plus its temporaries. Tiny chunks use very little memory, but
# give so many tasks that the scheduler's fixed cost per task starts to dominate.
# In between there is a broad sweet spot. The [dask
# docs](https://docs.dask.org/en/stable/array-chunks.html) suggest starting
# around 100 MB per chunk, with at least a few chunks per core.
#
# Chunk shape matters for files too. The LFP on DANDI is stored in chunks of
# about 5 s by all 384 channels, so reading a time window is cheap, while
# reading one channel for the whole session means reading every chunk.
#
# **Where next:** the [dask tutorial](https://tutorial.dask.org), `dask.dataframe`
# for pandas-like tables too big for memory, and the distributed scheduler's
# live dashboard (`from dask.distributed import Client; client = Client()`),
# which shows every task as it runs.
