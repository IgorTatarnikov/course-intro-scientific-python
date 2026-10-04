# %% [markdown]
# # Day 2 · SciPy
#
# **Block:** SciPy (25 min, about 5 of them slides)
#
# 1. **Curve fitting** (10 min): SPL Exercise 39, fit a yearly cycle to Alaska temperatures
# 2. **Image denoising with the FFT** (10 min): SPL Exercise 42, clean up the moon landing image
# 3. **Stretch:** statistical distributions (SPL Exercise 41) and 2-D minimisation
#    (SPL Exercise 40), then the chapter's summary exercises
#
# Adapted from [Scientific Python Lectures, SciPy: high-level scientific
# computing](https://lectures.scientific-python.org/intro/scipy/index.html) (CC BY 4.0).
#
# Before writing your own algorithm, **check whether SciPy already has it**.
# The [API reference](https://docs.scipy.org/doc/scipy/reference/) lists every submodule.

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# %% [markdown]
# ## 1. Curve fitting (SPL Exercise 39)
#
# The temperature extremes in Alaska for each month, starting in January, are (in °C):

# %%
temp_max = np.array([17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18])
temp_min = np.array([-62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58])
months = np.arange(12)

# %% [markdown]
# 1. Plot these temperature extremes.
# 2. Define a function that can describe both the minimum and the maximum
#    temperatures. Hint: it must have a period of one year (12 months), and it
#    needs a time offset.
# 3. Fit it to each series with `sp.optimize.curve_fit`.
# 4. Plot the fits on a fine time grid. Is the fit reasonable? If not, why not?
# 5. Is the time offset the same for the minimum and the maximum, within the fit's
#    accuracy? (Hint: `curve_fit` also returns a covariance matrix. The square roots
#    of its diagonal are the parameters' standard errors.)

# %% tags=["solution"]
def yearly_temps(times, avg, ampl, time_offset):
    return avg + ampl * np.cos((times + time_offset) * 2 * np.pi / 12)


res_max, cov_max = sp.optimize.curve_fit(yearly_temps, months, temp_max, p0=[20, 10, 0])
res_min, cov_min = sp.optimize.curve_fit(yearly_temps, months, temp_min, p0=[-40, 20, 0])

days = np.linspace(0, 12, num=365)
fig, ax = plt.subplots()
ax.plot(months, temp_max, "ro", label="max")
ax.plot(days, yearly_temps(days, *res_max), "r-")
ax.plot(months, temp_min, "bo", label="min")
ax.plot(days, yearly_temps(days, *res_min), "b-")
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (°C)")
ax.legend();

# %% tags=["solution"]
err_max = np.sqrt(np.diag(cov_max))
err_min = np.sqrt(np.diag(cov_min))
print(f"offset (max): {res_max[2]:.2f} ± {err_max[2]:.2f} months")
print(f"offset (min): {res_min[2]:.2f} ± {err_min[2]:.2f} months")

# %% [markdown] tags=["answer"]
# The fits follow the data closely: a cosine with a one-year period describes
# both series well. The offsets are 0.28 ± 0.10 months (max) and -0.16 ± 0.13
# months (min). Both amplitudes are negative, so the warmest point of each curve
# falls at month 6 minus the offset (counting January as 0): late June (5.7) for
# the maxima and early July (6.2) for the minima. The difference, about 0.4 months, is roughly 2.7
# combined standard errors, so the offsets are probably *not* the same. With
# only 12 points per series, though, do not read too much into it.
#
# Note that SPL's own solution divides by `times.max()` instead of 12. That makes
# the period depend on whichever time array you pass in: 11 months for `months`
# but 12 for `days`. It is a bug worth spotting!

# %% [markdown]
# ## 2. Image denoising with the FFT (SPL Exercise 42)
#
# `moonlanding.png` is heavily contaminated with periodic noise. Clean it up using
# the Fast Fourier Transform:
#
# 1. Load the image with `plt.imread` and display it in grey.
# 2. Find the 2-D FFT function in `sp.fft` and plot the spectrum (its absolute
#    value). Is it hard to see anything? Why? (Hint: try
#    `norm=matplotlib.colors.LogNorm(vmin=5)` in `imshow`.)
# 3. The spectrum has high- and low-frequency components. The noise is in the
#    **high**-frequency part, which for `fft2` output is the **middle** of the array.
#    Set those components to zero with slicing, keeping only, say, the first and last
#    10% of rows and columns.
# 4. Apply the inverse FFT and display the real part of the result.

# %%
from matplotlib.colors import LogNorm

im = plt.imread("../data/moonlanding.png").astype(float)
im.shape

# %% tags=["solution"]
fig, ax = plt.subplots()
ax.imshow(im, cmap="gray")
ax.set_title("Original image");

# %% tags=["solution"]
im_fft = sp.fft.fft2(im)

fig, ax = plt.subplots()
img = ax.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
fig.colorbar(img, ax=ax)
ax.set_title("Fourier transform (log scale)");

# %% [markdown] tags=["answer"]
# On a linear scale a few very large coefficients (the image's mean brightness
# and the lowest frequencies) dominate, and everything else looks black. A
# logarithmic colour scale shows the structure, including the bright spots of the
# periodic noise.

# %% tags=["solution"]
keep_fraction = 0.1
im_fft2 = im_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r * keep_fraction) : int(r * (1 - keep_fraction))] = 0
im_fft2[:, int(c * keep_fraction) : int(c * (1 - keep_fraction))] = 0

im_new = sp.fft.ifft2(im_fft2).real

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(np.abs(im_fft2), norm=LogNorm(vmin=5))
axes[0].set_title("Filtered spectrum")
axes[1].imshow(im_new, cmap="gray")
axes[1].set_title("Reconstructed image");

# %% [markdown] tags=["solution"]
# Zeroing the high frequencies is a crude **low-pass filter**: it blurs the image
# as well as removing the noise. `sp.ndimage.gaussian_filter(im, 4)` blurs in
# one line. The scikit-image block that comes next has many better filters.

# %% [markdown]
# ## 3. Stretch
#
# ### Statistical distributions (SPL Exercise 41)
#
# Draw 1000 random values from a gamma distribution with shape parameter 1 (hint:
# `sp.stats.gamma(1)` makes a "frozen" distribution, and its `.rvs` method draws
# values). Plot their histogram with the distribution's PDF on top. Then estimate
# the shape parameter back from the sample with `sp.stats.gamma.fit`.
#
# Extra: plot the cumulative distribution function and compute the variance.

# %% tags=["solution"]
rng = np.random.default_rng(0)
dist = sp.stats.gamma(1)
sample = dist.rvs(size=1000, random_state=rng)

x = np.linspace(0, sample.max(), 200)
fig, ax = plt.subplots()
ax.hist(sample, bins=40, density=True, alpha=0.5, label="sample")
ax.plot(x, dist.pdf(x), label="PDF")
ax.plot(x, dist.cdf(x), label="CDF")
ax.legend()

shape, loc, scale = sp.stats.gamma.fit(sample)
print(f"fitted shape: {shape:.2f}, variance: {dist.var():.2f}")

# %% [markdown]
# ### 2-D minimisation (SPL Exercise 40)
#
# The six-hump camelback function
#
# $$f(x, y) = (4 - 2.1x^2 + \tfrac{x^4}{3})x^2 + xy + (4y^2 - 4)y^2$$
#
# has several local minima. Find a global minimum (there are two, with the same
# value) and at least one other local minimum.
#
# * Restrict the search to $-2 < x < 2$ and $-1 < y < 1$.
# * Visualise it with `np.meshgrid` and `imshow`.
# * Try `sp.optimize.minimize` from (0, 0), then from other starting points. Does
#   it find the global minimum?
# * Try `sp.optimize.differential_evolution`.

# %%
def sixhump(x):
    return (
        (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
        + x[0] * x[1]
        + (-4 + 4 * x[1] ** 2) * x[1] ** 2
    )


# %% tags=["solution"]
xlim, ylim = [-2, 2], [-1, 1]
xg, yg = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 100))

res_local = sp.optimize.minimize(sixhump, x0=[0, 0])
res_global = sp.optimize.differential_evolution(sixhump, bounds=[xlim, ylim], seed=0)
print("minimize from (0, 0):", res_local.x.round(3), res_local.fun.round(3))
print("differential_evolution:", res_global.x.round(3), res_global.fun.round(3))

fig, ax = plt.subplots()
img = ax.imshow(sixhump([xg, yg]), extent=xlim + ylim, origin="lower")
fig.colorbar(img, ax=ax)
ax.scatter(*res_local.x, label="minimize from (0, 0)")
ax.scatter(*res_global.x, label="differential_evolution")
ax.legend();

# %% [markdown] tags=["solution"]
# Starting at (0, 0), which is a saddle point with zero gradient, `minimize` does
# not move at all. Starting elsewhere, it slides into whichever basin is nearest. A
# global method such as `differential_evolution` finds one of the two global minima
# at about (±0.09, ∓0.71), where f = -1.03.
#
# ### Summary exercises
#
# The [SPL summary
# exercises](https://lectures.scientific-python.org/intro/scipy/index.html#summary-exercises-on-scientific-computing)
# combine several submodules: statistical interpolation, non-linear fitting, and
# image processing.
