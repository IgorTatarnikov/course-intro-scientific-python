# %% [markdown]
# # Day 2 · Whirlwind tour: xarray
#
# **Block:** whirlwind tour, xarray (about 10 min: demo, then try it)
#
# **Problem it solves:** real N-D data has *named* dimensions (time, latitude,
# longitude; or z, channel, y, x) with *coordinates* along them. NumPy only knows
# axis 0, 1, 2 and positions. xarray keeps the names and coordinates attached,
# so you can write `ds.sel(time="2013-07")` instead of `air[181:212]`.
#
# The demo uses the NCEP air temperature reanalysis from the [xarray
# tutorial](https://tutorial.xarray.dev), reduced to daily means for 2013–2014
# over North America.

# %%
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("../data/air_temperature_daily.nc")
ds

# %% [markdown]
# ## Demo
#
# A `Dataset` holds one or more `DataArray`s that share coordinates. In Jupyter,
# click the icons in the output above to see the attributes and values.

# %%
air = ds["air"] - 273.15  # Kelvin to °C. Arithmetic keeps the coordinates.
air.attrs["units"] = "°C"
air

# %% [markdown]
# **Select by label**, not by position. `method="nearest"` snaps to the closest grid point:

# %%
point = air.sel(lat=50, lon=260, method="nearest")  # southern Canada
point.plot(figsize=(8, 3));

# %% [markdown]
# **Reduce over a named dimension.** No more remembering that time is axis 0:

# %%
air.mean(dim="time").plot(cmap="RdBu_r", figsize=(8, 4));

# %% [markdown]
# **Group by** a property of a coordinate, here the season of each date:

# %%
seasonal = air.groupby("time.season").mean().sel(season=["DJF", "MAM", "JJA", "SON"])
seasonal.plot(col="season", cmap="RdBu_r", col_wrap=4, figsize=(14, 3));

# %% [markdown]
# Slicing by time understands date strings:

# %%
air.sel(time=slice("2013-07-01", "2013-07-31")).mean(dim=["lat", "lon"]).plot(figsize=(8, 3));

# %% [markdown]
# ## Try it
#
# 1. Compare the daily temperature in 2013 at the grid points nearest
#    **Chicago** (41.9° N, 87.6° W) and **Miami** (25.8° N, 80.2° W) on one plot.
#    Careful: this dataset's longitudes run from 200 to 330 **degrees east**,
#    so 87.6° W is 360 − 87.6 = 272.4° E.
# 2. Which city has the larger temperature range over the year?
# 3. Bonus: write the same selection for Chicago **without** xarray, using NumPy
#    indexing on `air.values`. Which version would you rather read in six months?

# %% tags=["solution"]
cities = {"Chicago": (41.9, 360 - 87.6), "Miami": (25.8, 360 - 80.2)}

fig, ax = plt.subplots(figsize=(9, 3.5))
for name, (lat, lon) in cities.items():
    series = air.sel(lat=lat, lon=lon, method="nearest").sel(time="2013")
    series.plot(ax=ax, label=name)
    print(f"{name}: range {float(series.max() - series.min()):.1f} °C")
ax.set_title("Daily mean temperature, 2013")
ax.legend();

# %% tags=["solution"]
# Bonus: the NumPy way. You need to know the axis order and look up the indices yourself.
import numpy as np

i_lat = np.abs(ds["lat"].values - 41.9).argmin()
i_lon = np.abs(ds["lon"].values - (360 - 87.6)).argmin()
i_2013 = ds["time"].dt.year.values == 2013
chicago_numpy = air.values[i_2013, i_lat, i_lon]
np.allclose(chicago_numpy, air.sel(lat=41.9, lon=272.4, method="nearest").sel(time="2013"))

# %% [markdown] tags=["answer"]
# Chicago swings over a much larger range than Miami, which stays warm all year.
# The NumPy version works, but it depends on knowing that time is axis 0 and
# latitude axis 1, and on finding indices by hand. The xarray line says what it
# means.
#
# **Where next:** the [xarray tutorial](https://tutorial.xarray.dev), and
# `xr.open_dataset(..., chunks={})`, which hands the computation to **dask**
# (the next notebook).
