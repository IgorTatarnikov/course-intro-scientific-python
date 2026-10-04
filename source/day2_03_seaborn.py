# %% [markdown]
# # Day 2 · seaborn
#
# **Block:** seaborn (15 min)
#
# The first part is a **live demo** (6 min): follow along, run the cells and
# change things. The demo mirrors the Python Data Science Handbook's
# [Visualization with
# Seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html),
# but uses the tidy tables you made in the pandas block. Then it is your turn:
#
# 1. **The psychometric curve** (5 min), straight from the trials table, with
#    confidence intervals
# 2. **Stretch:** one plot of your own
#
# The [seaborn tutorial](https://seaborn.pydata.org/tutorial.html) and
# [example gallery](https://seaborn.pydata.org/examples/index.html) are the best
# places to look things up.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def read_table(group):
    """Read one group of ../data/ibl_session.h5 into a DataFrame (as in the pandas block)."""
    with h5py.File("../data/ibl_session.h5") as f:
        columns = {}
        for name, values in f[group].items():
            values = values[:]
            if values.dtype.kind == "S":
                values = values.astype(str)
            columns[name] = values
    return pd.DataFrame(columns)


trials = read_table("trials")
units = read_table("units")
trials["chose_right"] = trials["choice"] == 1

# %% [markdown]
# seaborn wants **tidy** tables: one row per observation and one column per
# variable. `trials` and `units` already are. The cell below rebuilds
# `rates_long` from part 4 of the pandas block (the mean firing rate of each
# unit in each time bin, one row per unit and bin), and adds each unit's brain
# area to it with `merge`.

# %%
with h5py.File("../data/ibl_session.h5") as f:
    spike_counts = f["spike_counts/data"][:]
    spike_unit = f["spike_counts/unit"][:]
    bin_start = f["spike_counts/bin_start_s"][:]

rates = pd.DataFrame(
    spike_counts.mean(axis=1) / 0.05,  # spikes per second
    index=pd.Index(spike_unit, name="unit"),
    columns=(bin_start + 0.025).round(3),  # the centre of each bin
)
rates_long = rates.reset_index().melt(id_vars="unit", var_name="time_s", value_name="rate_Hz")
rates_long = rates_long.merge(units[["unit", "area"]], on="unit")  # add each unit's area
rates_long.head()

# %% [markdown]
# ## Demo
#
# ### Relationships: `relplot`
#
# Each argument maps a **column** to a visual property. The legend is built for us.

# %%
g = sns.relplot(
    data=units,
    x="amplitude_uV",
    y="depth_um",
    hue="area",
    size="firing_rate_Hz",
    sizes=(5, 300),
    alpha=0.6,
    height=6,
    aspect=0.9,
)
g.set(xscale="log", xlabel="Spike amplitude (µV, log scale)", ylabel="Height above probe tip (µm)");

# %% [markdown]
# The same plot in plain Matplotlib needs a loop over areas, manual colours
# and a hand-made size legend. That is the main reason to reach for seaborn when
# your data is already in a DataFrame.
#
# ### Averages, with uncertainty: `lineplot`
#
# Many units share each time bin. seaborn **aggregates** them (mean by default)
# and shades a 95% confidence interval: here one line per area, each the mean
# over its units, with a band across units.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(data=rates_long, x="time_s", y="rate_Hz", hue="area", ax=ax)
ax.axvline(0, color="grey", linestyle="--")
ax.set(xlabel="Time from stimulus (s)", ylabel="Firing rate (spikes/s)");

# %% [markdown]
# ### Distributions: `histplot`, `kdeplot`, `boxplot`, `violinplot`

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(
    data=units[units["spike_width_ms"] > 0],
    x="spike_width_ms",
    hue="label",
    multiple="stack",
    bins=20,
    ax=axes[0],
)
sns.boxplot(
    data=trials.assign(strength=trials["contrast"].abs()),
    x="strength",
    y="response_time_s",
    log_scale=True,
    ax=axes[1],
)
axes[1].set(xlabel="Contrast (%, either side)", ylabel="Response time (s, log scale)");

# %% [markdown]
# Spike widths have two humps: narrow spikes (often inhibitory interneurons)
# and broad ones (often excitatory pyramidal cells). The response times are
# skewed, with a long tail of slow trials, so a log scale shows them better.
#
# ### Small multiples: one panel per category with `col=`

# %%
sns.displot(
    data=units,
    x="firing_rate_Hz",
    col="area",
    col_wrap=4,
    height=2.5,
    bins=15,
);

# %% [markdown]
# Every seaborn figure is still Matplotlib underneath: axes-level functions
# (`lineplot`, `boxplot`) take `ax=` and return an `Axes`; figure-level
# functions (`relplot`, `displot`, `lmplot`, `catplot`) return a `FacetGrid` `g`,
# with `g.figure` and `g.axes`. Customise and save it with the Matplotlib from
# the previous block.
#
# ## Your turn
#
# ### 1. The psychometric curve, with confidence intervals
#
# The mean of `chose_right` at each contrast is the psychometric curve, so
# `sns.lineplot` can draw it straight from `trials`, with no `groupby`: it
# aggregates the trials at each contrast itself and adds a confidence interval.
#
# 1. Plot `chose_right` against `contrast`, one line per block (`hue=`), with a
#    marker at each contrast. Label the axes.
# 2. Watch out: `probability_left` is a number, so as a `hue` seaborn gives it a
#    *continuous* colour scale, and the three blocks are hard to tell apart. Make
#    a `block` column that holds it as a string, so it becomes a category with
#    distinct colours, and plot again.
# 3. Where are the confidence intervals widest, and why?

# %% tags=["solution"]
trials["block"] = trials["probability_left"].astype(str)

fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(data=trials, x="contrast", y="chose_right", hue="block", marker="o", ax=ax)
ax.set(xlabel="Contrast (%; negative = left)", ylabel="Fraction of rightward choices");

# %% [markdown] tags=["answer"]
# Widest in the 0.5 block, which has only 10 trials at each contrast, and at
# the contrasts that are rare in a block, such as -100% in right blocks (5
# trials). The interval shrinks with the number of trials behind each point,
# which the plain curve from the NumPy block did not show.

# %% [markdown]
# ### 2. Stretch: one plot of your own
#
# Use `trials`, `units` or `rates_long` to make one seaborn plot that answers a
# question **you** find interesting. Give it proper axis labels and save it to a
# file. Ideas:
#
# * Does the mouse respond faster on correct trials than on wrong ones, at every
#   contrast? (`catplot` with `kind="box"` or `kind="violin"` and `hue="correct"`)
# * Do narrow-spiking units fire faster than broad-spiking ones? (make a column
#   that splits `spike_width_ms` at 0.4 ms first)
# * Does the mouse get better or worse during the session? (a rolling mean of
#   `correct` against trial number, with `.rolling(50).mean()`)

# %% tags=["solution"]
# One possible answer: response times on correct and wrong trials, by stimulus strength.
quick = trials[trials["response_time_s"] < 5].copy()
quick["strength"] = quick["contrast"].abs()
quick["outcome"] = np.where(quick["correct"], "correct", "wrong")
g = sns.catplot(
    data=quick,
    x="strength",
    y="response_time_s",
    hue="outcome",
    kind="violin",
    split=True,
    inner="quart",
    density_norm="width",  # every violin equally wide, however many trials it has
    cut=0,  # do not draw the density beyond the observed values
    log_scale=True,
    height=4,
    aspect=2,
)
g.set_axis_labels("Contrast (%, either side)", "Response time (s, log scale)")
g.figure.suptitle("Wrong answers are slower, most of all on easy trials", y=1.03)
g.savefig("response_times_by_outcome.png", dpi=150)
