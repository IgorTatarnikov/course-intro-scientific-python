# %% [markdown]
# # Day 2 · Matplotlib
#
# **Block:** Matplotlib (45 min, including slides and live coding)
#
# 1. **Simple plot** (12 min): build a sine/cosine figure step by step
# 2. **Scatter and imshow** (7 min): SPL Exercises 28 and 31
# 3. **Framing a Face** (10 min): crop and mask an image, held over from Day 1's NumPy
# 4. **Stretch:** hare and lynx (SPL Exercise 23, part 6), multiple subplots,
#    gapminder plots, and more SPL plot types
#
# Adapted from [Scientific Python Lectures, Matplotlib:
# plotting](https://lectures.scientific-python.org/intro/matplotlib/index.html) and
# [NumPy exercises](https://lectures.scientific-python.org/intro/numpy/exercises.html)
# (CC BY 4.0). We use the object-oriented style (`fig, ax = plt.subplots()`)
# throughout; SPL mostly uses the `plt.` functions, and the [Matplotlib
# docs](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html)
# explain the difference.

# %%
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Simple plot
#
# We want the cosine and sine functions on one plot, starting from the defaults
# and improving it step by step. Each step changes the same `fig` and `ax`; end each
# cell with `fig` so Jupyter shows the updated figure.

# %%
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)

# %% [markdown]
# **Step 1: defaults.** Create a figure and axes with `plt.subplots`, and plot `C`
# and `S` against `X`.

# %% tags=["solution"]
fig, ax = plt.subplots()
ax.plot(X, C)
ax.plot(X, S);

# %% [markdown]
# **Step 2: colours and line widths.** Make a new figure of size 10 x 6 inches
# (`figsize`). Plot the cosine in blue and the sine in red, both with a line width of
# 2.5. Add `label="cosine"` and `label="sine"`: you will need them for the legend.

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
ax.plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine");

# %% [markdown]
# **Step 3: limits.** The limits are a bit tight. Set the x and y limits to 1.2
# times the data's minimum and maximum (`ax.set_xlim`, `ax.set_ylim`).

# %% tags=["solution"]
ax.set_xlim(X.min() * 1.2, X.max() * 1.2)
ax.set_ylim(C.min() * 1.2, C.max() * 1.2)
fig

# %% [markdown]
# **Step 4: ticks and tick labels.** Put x ticks only at $-\pi, -\pi/2, 0, \pi/2, \pi$
# and y ticks at $-1, 0, 1$. Label the x ticks with LaTeX strings such as
# `r"$-\pi$"`. (`ax.set_xticks` takes both positions and `labels=`.)

# %% tags=["solution"]
ax.set_xticks(
    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    labels=[r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"],
)
ax.set_yticks([-1, 0, +1], labels=[r"$-1$", r"$0$", r"$+1$"])
fig

# %% [markdown]
# **Step 5: spines.** Spines are the lines around the plotting area. Hide the top
# and right ones, and move the bottom and left ones so they cross at the origin
# (`ax.spines["left"].set_position(("data", 0))`).

# %% tags=["solution"]
ax.spines[["top", "right"]].set_visible(False)
ax.spines["bottom"].set_position(("data", 0))
ax.spines["left"].set_position(("data", 0))
fig

# %% [markdown]
# **Step 6: legend.** Add a legend in the upper left corner.

# %% tags=["solution"]
ax.legend(loc="upper left", frameon=False)
fig

# %% [markdown]
# **Step 7 (optional, skip it if you are short of time): annotate.** At $t = 2\pi/3$, draw a dashed vertical line from 0 to each
# curve, mark each point with `ax.scatter`, and label it with `ax.annotate`, for
# example $\cos(2\pi/3) = -1/2$ with an arrow. See the [annotation
# guide](https://matplotlib.org/stable/users/explain/text/annotations.html).

# %% tags=["solution"]
t = 2 * np.pi / 3
ax.plot([t, t], [0, np.cos(t)], color="blue", linewidth=2.5, linestyle="--")
ax.scatter([t], [np.cos(t)], 50, color="blue")
ax.annotate(
    r"$\cos(\frac{2\pi}{3})=-\frac{1}{2}$",
    xy=(t, np.cos(t)),
    xycoords="data",
    xytext=(-90, -50),
    textcoords="offset points",
    fontsize=16,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
ax.plot([t, t], [0, np.sin(t)], color="red", linewidth=2.5, linestyle="--")
ax.scatter([t], [np.sin(t)], 50, color="red")
ax.annotate(
    r"$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$",
    xy=(t, np.sin(t)),
    xycoords="data",
    xytext=(+10, +30),
    textcoords="offset points",
    fontsize=16,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
fig

# %% [markdown]
# **Step 8: save it.** Save the figure as `sine_cosine.png` at 150 dpi and as
# `sine_cosine.pdf`. Open both files. What happens when you zoom in on each?

# %% tags=["solution"]
fig.savefig("sine_cosine.png", dpi=150, bbox_inches="tight")
fig.savefig("sine_cosine.pdf", bbox_inches="tight")

# %% [markdown] tags=["solution"]
# The PNG is a grid of pixels and goes blocky when you zoom in. The PDF is a
# **vector** format and stays sharp, which is what journals usually want for
# line plots. Use `dpi=300` or more for raster images in print.

# %% [markdown]
# ## 2. Other types of plots
#
# ### Scatter (SPL Exercise 28)
#
# Starting from the code below, reproduce the figure in
# [SPL's scatter plot section](https://lectures.scientific-python.org/intro/matplotlib/index.html#scatter-plots),
# paying attention to marker **size**, **colour** and **transparency**: large,
# half-transparent markers, coloured by the angle of each point, $\arctan2(Y, X)$,
# with axis limits of ±1.5 and no ticks.

# %%
n = 1024
X = rng.normal(0, 1, n)
Y = rng.normal(0, 1, n)

fig, ax = plt.subplots()
ax.scatter(X, Y);

# %% tags=["solution"]
T = np.arctan2(Y, X)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X, Y, s=75, c=T, alpha=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([])
ax.set_yticks([]);

# %% [markdown]
# ### Imshow (SPL Exercise 31)
#
# Starting from the code below, reproduce the
# [SPL imshow figure](https://lectures.scientific-python.org/intro/matplotlib/index.html#imshow),
# paying attention to the **colormap** (`"bone"`), the image **interpolation**
# (blocky pixels, not smoothed), the **origin** (row 0 at the bottom) and a **colorbar**.

# %%
def f(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2) - y**2)


n = 10
x = np.linspace(-3, 3, int(3.5 * n))
y = np.linspace(-3, 3, int(3.0 * n))
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots()
ax.imshow(Z);

# %% tags=["solution"]
fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation="nearest", cmap="bone", origin="lower")
fig.colorbar(im, ax=ax, shrink=0.92)
ax.set_xticks([])
ax.set_yticks([]);

# %% [markdown] tags=["solution"]
# `origin="lower"` puts row 0 at the bottom, like a graph. The default,
# `origin="upper"`, puts row 0 at the top, like a matrix or a photograph. Images
# from microscopes and cameras usually want the default.

# %% [markdown]
# ## 3. Framing a Face (from SPL's NumPy exercises)
#
# An image is just a 2-D (grey) or 3-D (colour) NumPy array. Load the raccoon
# face and convert it to grey by averaging the red, green and blue channels:

# %%
face_rgb = plt.imread("../data/face.png")
face = face_rgb.mean(axis=2)
face.shape, face.dtype, face.min(), face.max()

# %% [markdown]
# **a.** Display `face` with `ax.imshow`. Why are the colours odd? Fix them with
# `cmap="gray"`.

# %% tags=["solution"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(face)  # default colormap: viridis
axes[1].imshow(face, cmap="gray")
for ax in axes:
    ax.set_axis_off()

# %% [markdown] tags=["solution"]
# A 2-D array has no colour of its own. Matplotlib maps its values through a
# colormap, which is `viridis` by default.

# %% [markdown]
# **b. Narrow centring.** Make `crop_face` by removing 100 pixels from every border,
# and display it.

# %% tags=["solution"]
crop_face = face[100:-100, 100:-100]
fig, ax = plt.subplots()
ax.imshow(crop_face, cmap="gray")
crop_face.shape

# %% [markdown]
# **c. Frame the face.** Make a black circular "locket" around the face. The face's
# centre is around (x, y) = (660, 300). Set every pixel **further than 230 pixels**
# from the centre to 0.
#
# Hint: `y, x = np.ogrid[0:sy, 0:sx]` gives a column of row indices and a row of
# column indices. Broadcasting them against each other gives a full 2-D mask.
# Work on a **copy** of `face`, so the original stays intact.

# %% tags=["solution"]
locket = face.copy()
sy, sx = locket.shape
y, x = np.ogrid[0:sy, 0:sx]
centerx, centery = 660, 300
mask = ((y - centery) ** 2 + (x - centerx) ** 2) > 230**2
locket[mask] = 0

fig, ax = plt.subplots()
ax.imshow(locket, cmap="gray")
ax.set_axis_off()

# %% [markdown]
# **d. Follow-up.** Change the circle to an ellipse that is wider than it is tall.

# %% tags=["solution"]
ellipse = face.copy()
a, b = 330, 230  # half-width, half-height
mask = ((x - centerx) / a) ** 2 + ((y - centery) / b) ** 2 > 1
ellipse[mask] = 0

fig, ax = plt.subplots()
ax.imshow(ellipse, cmap="gray")
ax.set_axis_off()

# %% [markdown]
# ## 4. Stretch
#
# ### Hare and lynx (SPL Exercise 23, part 6)
#
# Back to `populations.txt` from Day 1. Plot the change in hare population
# (`np.gradient`) and the number of lynxes on the same axes, over time. Then check
# their correlation with `np.corrcoef`. What story do the two curves tell?

# %%
data = np.loadtxt("../data/populations.txt")
year, hares, lynxes, carrots = data.T

# %% tags=["solution"]
hare_grad = np.gradient(hares)
print("corr(d hares/dt, lynxes):", np.corrcoef(hare_grad, lynxes)[0, 1].round(2))

fig, ax = plt.subplots()
ax.plot(year, hare_grad, label="change in hares per year")
ax.plot(year, lynxes, label="lynxes")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("year")
ax.legend();

# %% [markdown] tags=["solution"]
# The correlation is about -0.9: hare numbers fall fastest when there are most
# lynxes. It is the classic predator-prey cycle.

# %% [markdown]
# ### Multiple subplots (SPL Exercise 35)
#
# Reproduce the [SPL multi-plot layout](https://lectures.scientific-python.org/intro/matplotlib/index.html#multi-plots):
# one wide panel on top, three small panels below. SPL uses `plt.subplot`. Rebuild it
# with `plt.subplot_mosaic`, as in the Handbook's [Multiple
# Subplots](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
# chapter, which shows `GridSpec`, the older equivalent. Then put a plot of your
# choice in each panel.

# %% tags=["solution"]
fig, axd = plt.subplot_mosaic([["top", "top", "top"], ["a", "b", "c"]], figsize=(8, 5), layout="constrained")
axd["top"].plot(year, hares, label="hares")
axd["top"].plot(year, lynxes, label="lynxes")
axd["top"].legend()
axd["a"].hist(hares, bins=8)
axd["b"].scatter(hares, lynxes, s=10)
axd["c"].imshow(face[::8, ::8], cmap="gray")
axd["c"].set_axis_off()

# %% [markdown]
# ### gapminder: minima and maxima
#
# From Software Carpentry's [Plotting](https://swcarpentry.github.io/python-novice-gapminder/09-plotting.html)
# episode: plot the **minimum** and the **maximum** GDP per capita over time for
# the countries of Europe, on one set of axes, with a legend. (Hint: pandas
# DataFrames and Series have a `.plot()` method that accepts `ax=`.)

# %%
import pandas as pd

data_europe = pd.read_csv("../data/gapminder_gdp_europe.csv", index_col="country")

# %% tags=["solution"]
years = data_europe.columns.str.removeprefix("gdpPercap_").astype(int)
fig, ax = plt.subplots()
ax.plot(years, data_europe.min(), label="min")
ax.plot(years, data_europe.max(), label="max")
ax.set_xlabel("year")
ax.set_ylabel("GDP per capita")
ax.legend();

# %% [markdown]
# ### More plot types
#
# SPL Exercises 27 (`fill_between`), 29 (bar labels), 30 (contours), 32 (pie),
# 33 (quiver) and 36 (polar) in [Other Types of
# Plots](https://lectures.scientific-python.org/intro/matplotlib/index.html#other-types-of-plots-examples-and-exercises).
# Skip Exercise 37: its starter code uses `Axes3D(fig)`, which current Matplotlib
# no longer supports that way. Use `fig.add_subplot(projection="3d")` instead.
#
# Reference: the Handbook's chapters on [line
# plots](https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html),
# [error bars](https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html)
# and [colorbars](https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html).
