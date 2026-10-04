# %% [markdown]
# # Day 2 · SciPy
#
# **Block:** SciPy (25 min, including slides and live coding)
#
# 1. **Curve fitting** (8 min): after SPL Exercise 39, fit the mouse's
#    psychometric curve in each block of trials
# 2. **Filtering with the FFT** (8 min): after SPL Exercise 42, remove the mains
#    hum from the LFP
# 3. **Stretch:** which distribution fits the response times (after SPL
#    Exercise 41), 2-D minimisation (SPL Exercise 40), then the chapter's
#    summary exercises
#
# We keep using the recording from Day 1. The exercises are adapted from
# [Scientific Python Lectures, SciPy: high-level scientific
# computing](https://lectures.scientific-python.org/intro/scipy/index.html) (CC
# BY 4.0), which fits Alaska's temperatures and cleans up a photograph instead.
#
# Before writing your own algorithm, **check whether SciPy already has it**.
# The [API reference](https://docs.scipy.org/doc/scipy/reference/) lists every submodule.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

with h5py.File("../data/ibl_session.h5") as f:
    contrast = f["trials/contrast"][:]
    choice = f["trials/choice"][:]
    probability_left = f["trials/probability_left"][:]
    rt = f["trials/response_time_s"][:]
    lfp_uv = f["lfp/data"][:] * f["lfp"].attrs["uV_per_count"]
    sampling_rate = f["lfp"].attrs["sampling_rate_Hz"]

# %% [markdown]
# ## 1. Curve fitting (after SPL Exercise 39)
#
# On Day 1 you computed the **psychometric curve**: the fraction of rightward
# choices at each contrast. The task switches between blocks in which the
# stimulus is more often on the left (`probability_left` 0.8) or on the right
# (0.2). Here is the curve for each kind of block:

# %%
levels = np.unique(contrast)
right_block = probability_left == 0.2
left_block = probability_left == 0.8
frac_right_rb = np.array([(choice[right_block & (contrast == c)] == 1).mean() for c in levels])
frac_right_lb = np.array([(choice[left_block & (contrast == c)] == 1).mean() for c in levels])
levels, frac_right_rb.round(2), frac_right_lb.round(2)

# %% [markdown]
# 1. Plot both curves against contrast.
# 2. Write a function `psychometric(c, bias, width, lapse)` that can describe
#    both. A common choice is a cumulative Gaussian, which rises from `lapse` to
#    `1 - lapse`, is centred on `bias` and rises over about `width` (in %
#    contrast):
#
#    $$p(c) = \text{lapse} + (1 - 2\,\text{lapse})\, \tfrac{1}{2}\left(1 + \text{erf}\!\left(\frac{c - \text{bias}}{\sqrt{2}\,\text{width}}\right)\right)$$
#
#    `sp.special.erf` is the error function.
# 3. Fit it to each curve with `sp.optimize.curve_fit`, starting from
#    `p0=[0, 10, 0.05]`.
# 4. Plot the fits on a fine contrast grid. Is the fit reasonable?
# 5. Is the `bias` the same in the two kinds of block, within the fit's
#    accuracy? (Hint: `curve_fit` also returns a covariance matrix. The square
#    roots of its diagonal are the parameters' standard errors.) What does the
#    answer say about the mouse?

# %% tags=["solution"]
def psychometric(c, bias, width, lapse):
    return lapse + (1 - 2 * lapse) * 0.5 * (1 + sp.special.erf((c - bias) / (np.sqrt(2) * width)))


res_rb, cov_rb = sp.optimize.curve_fit(psychometric, levels, frac_right_rb, p0=[0, 10, 0.05])
res_lb, cov_lb = sp.optimize.curve_fit(psychometric, levels, frac_right_lb, p0=[0, 10, 0.05])

fine = np.linspace(-100, 100, 401)
fig, ax = plt.subplots()
ax.plot(levels, frac_right_rb, "ro", label="right block (p left = 0.2)")
ax.plot(fine, psychometric(fine, *res_rb), "r-")
ax.plot(levels, frac_right_lb, "bo", label="left block (p left = 0.8)")
ax.plot(fine, psychometric(fine, *res_lb), "b-")
ax.set_xlabel("Contrast (%; negative = left)")
ax.set_ylabel("Fraction of rightward choices")
ax.legend();

# %% tags=["solution"]
err_rb = np.sqrt(np.diag(cov_rb))
err_lb = np.sqrt(np.diag(cov_lb))
for name, res, err in [("right block", res_rb, err_rb), ("left block", res_lb, err_lb)]:
    print(f"{name}: bias {res[0]:.1f} ± {err[0]:.1f} %, width {res[1]:.1f} ± {err[1]:.1f} %, lapse {res[2]:.2f}")
difference = res_lb[0] - res_rb[0]
print(f"difference: {difference:.1f} % = {difference / np.hypot(err_rb[0], err_lb[0]):.1f} combined standard errors")

# %% [markdown] tags=["answer"]
# The fits follow the points closely; the curves only differ in the middle,
# where the stimulus is hard to see. In right blocks the bias is about 0%; in left
# blocks about 6%, roughly four combined standard errors higher, so the shift is
# real. A positive bias means the mouse needs more contrast on the right
# before it chooses right: when the stimulus has mostly been on the left, it
# guesses left when unsure. The mouse has learned the blocks' statistics and
# uses them as a prior.
#
# The fit treats every contrast equally, although some have only 5 trials in a
# block and others 50. Passing `sigma=` (the standard error of each fraction)
# to `curve_fit` weights the points properly; a fit to the individual choices by
# maximum likelihood is better still.

# %% [markdown]
# ## 2. Filtering with the FFT (after SPL Exercise 42)
#
# The recording was made in the United States, where mains electricity
# alternates at **60 Hz**. Every cable in the room radiates it, and some ends up
# in the LFP. Remove it with the Fast Fourier Transform:
#
# 1. Take `trace`, channel 200 of `lfp_uv`, and plot its first 0.5 s.
# 2. Find the FFT for real signals in `sp.fft` (`rfft`), and the matching
#    frequencies (`rfftfreq`, which needs the time between samples). Plot the
#    spectrum (the absolute value) against frequency. Is it hard to see
#    anything? Why? (Hint: `ax.set_yscale("log")`.)
# 3. Find the mains hum in the spectrum. Set the components within 1 Hz of
#    60 Hz **and of its harmonics**, 120 and 180 Hz, to zero, using a mask on the
#    frequencies.
# 4. Apply the inverse FFT (`irfft`, with `n=` the length of `trace`), and plot
#    what you **removed**, `trace - cleaned`, for the first 0.2 s. What does it
#    look like?

# %%
trace = lfp_uv[:, 200]
time = np.arange(trace.size) / sampling_rate

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(time[:250], trace[:250])
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (µV)");

# %% tags=["solution"]
spectrum = sp.fft.rfft(trace)
freqs = sp.fft.rfftfreq(trace.size, d=1 / sampling_rate)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(freqs, np.abs(spectrum))
axes[0].set_title("linear scale")
axes[1].plot(freqs, np.abs(spectrum))
axes[1].set_yscale("log")
axes[1].set_title("log scale")
for ax in axes:
    ax.set_xlabel("frequency (Hz)")

# %% [markdown] tags=["answer"]
# On a linear scale the slow frequencies, below a few hertz, dominate (the LFP is
# mostly slow waves) and everything else looks flat. On a log scale the spectrum
# falls steadily with frequency, with sharp spikes at 60, 120 and 180 Hz: the
# mains hum and its harmonics. The FFT of a real signal is symmetric, so `rfft`
# keeps only the positive frequencies, up to half the sampling rate (250 Hz).

# %% tags=["solution"]
hum = (np.abs(freqs - 60) < 1) | (np.abs(freqs - 120) < 1) | (np.abs(freqs - 180) < 1)
filtered = spectrum.copy()
filtered[hum] = 0
cleaned = sp.fft.irfft(filtered, n=trace.size)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(freqs, np.abs(spectrum), label="original")
axes[0].plot(freqs, np.abs(filtered), label="filtered")
axes[0].set_yscale("log")
axes[0].set_xlim(40, 200)
axes[0].set_xlabel("frequency (Hz)")
axes[0].legend()
axes[1].plot(time[:100], (trace - cleaned)[:100])
axes[1].set_xlabel("time (s)")
axes[1].set_title("what was removed (µV)");

# %% [markdown] tags=["solution"]
# What was removed repeats 12 times in 0.2 s: 60 Hz, as expected, peaking at
# about 15 µV. Each cycle has a second, smaller bump: that is the 120 Hz
# harmonic. Zeroing FFT bins like this is a crude **notch
# filter**: fine for a quick look, but it can ring at the edges of the
# recording. `sp.signal.iirnotch` with `sp.signal.filtfilt` is the usual tool.
# `rfft(lfp_uv, axis=0)` would clean all 384 channels in one call.

# %% [markdown]
# ## 3. Stretch
#
# ### Which distribution fits the response times? (after SPL Exercise 41)
#
# Response times are positive and skewed, with a long tail. Take the response
# times below 5 s (the others are trials where the mouse was not engaged). Fit
# a gamma distribution and a lognormal distribution to them with
# `sp.stats.gamma.fit` and `sp.stats.lognorm.fit`, fixing the location at 0
# (`floc=0`). Plot their PDFs on top of the histogram. Which fits better? Compare
# the total log-likelihood, `dist.logpdf(sample, *params).sum()`, too: higher is
# better.
#
# Extra: plot the cumulative distribution function of the better fit, and use
# its `.ppf` to find the response time that 90% of trials beat.

# %% tags=["solution"]
sample = rt[rt < 5]
params_gamma = sp.stats.gamma.fit(sample, floc=0)
params_lognorm = sp.stats.lognorm.fit(sample, floc=0)

x = np.linspace(0.01, 2, 300)
fig, ax = plt.subplots()
ax.hist(sample, bins=np.linspace(0, 2, 60), density=True, alpha=0.5, label="response times")
ax.plot(x, sp.stats.gamma.pdf(x, *params_gamma), label="gamma")
ax.plot(x, sp.stats.lognorm.pdf(x, *params_lognorm), label="lognormal")
ax.set_xlabel("response time (s)")
ax.legend()

for name, dist, params in [("gamma", sp.stats.gamma, params_gamma), ("lognormal", sp.stats.lognorm, params_lognorm)]:
    print(f"{name:10} log-likelihood {dist.logpdf(sample, *params).sum():.1f}")
print(f"90% of responses are faster than {sp.stats.lognorm.ppf(0.9, *params_lognorm):.2f} s")

# %% [markdown] tags=["solution"]
# The lognormal fits much better: it has the sharp peak around 0.3 s and the long
# tail, and its log-likelihood is about 110 higher. A lognormal means that the
# **logarithm** of the response time is normally distributed, which is why the
# log scale suited these data in the seaborn block.
#
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
