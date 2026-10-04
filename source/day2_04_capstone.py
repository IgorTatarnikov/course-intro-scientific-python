# %% [markdown]
# # Day 2 · Capstone: one figure from two days
#
# **Block:** capstone (15 min)
#
# Yesterday you checked your results with quick plots. Today you know enough
# Matplotlib to make a figure you could put in a paper or a talk. Build **one
# figure with two panels** from the work of both days:
#
# * **A.** The psychometric curve in each kind of block: the data (pandas) and
#   the model's curves (scikit-learn)
# * **B.** The LFP before and after filtering (SciPy)
#
# The cells below recompute everything the figure needs, so you can start here
# whatever you finished yesterday.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.linear_model import LogisticRegression


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
trials["chose_right"] = trials["choice"] == 1

with h5py.File("../data/ibl_session.h5") as f:
    lfp_uv = f["lfp/data"][:] * f["lfp"].attrs["uV_per_count"]
    sampling_rate = f["lfp"].attrs["sampling_rate_Hz"]

# %% [markdown]
# **The data for panel A** (pandas block, part 3): the fraction of rightward
# choices at each contrast, one column per block.

# %%
psychometric = trials.groupby(["contrast", "probability_left"])["chose_right"].mean().unstack()
psychometric.round(2)

# %% [markdown]
# **The model for panel A** (scikit-learn stretch): logistic regression on
# `tanh(contrast / 25)` and the block, fitted on all the trials. `p_right` gives
# its probability of a rightward choice at any contrast, in a block with the
# given `probability_left`.

# %%
def features(contrast, probability_left):
    """The model's X: one row per contrast; probability_left can be one value or one per row."""
    return np.column_stack([np.tanh(contrast / 25), np.broadcast_to(probability_left, contrast.shape)])


model = LogisticRegression().fit(
    features(trials["contrast"].to_numpy(), trials["probability_left"].to_numpy()),
    trials["chose_right"],
)


def p_right(contrast, probability_left):
    return model.predict_proba(features(contrast, probability_left))[:, 1]


grid = np.linspace(-100, 100, 201)
p_right(np.array([-25.0, 0.0, 25.0]), 0.2).round(2)

# %% [markdown]
# **The data for panel B** (SciPy block, part 2): channel 200, raw and with a
# 1-30 Hz band-pass filter, for the first 2 s.

# %%
time = np.arange(1000) / sampling_rate
raw = lfp_uv[:1000, 200]
sos = sp.signal.butter(4, [1, 30], btype="bandpass", fs=sampling_rate, output="sos")
slow = sp.signal.sosfiltfilt(sos, lfp_uv[:, 200])[:1000]

# %% [markdown]
# ## Your figure
#
# Build the figure step by step, as in part 1 of the Matplotlib block. Aim for:
#
# 1. Two panels side by side: `plt.subplots(1, 2, figsize=(11, 4),
#    layout="constrained")`, or `plt.subplot_mosaic([["A", "B"]], ...)`.
# 2. **Panel A:** the data as points and the model as lines, in the **same
#    colour for the same block**, for the 0.2 and 0.8 blocks. (Hint: each
#    column of `psychometric` is one block; `psychometric.index` holds the
#    contrasts.) A legend that says which block is which, in words.
# 3. **Panel B:** the raw trace in light grey and the filtered one on top in a
#    strong colour, with a legend.
# 4. Axis labels **with units** on every axis, and the top and right spines
#    hidden.
# 5. Panel letters: `ax.set_title("A", loc="left", fontweight="bold")`.
# 6. Save it as `capstone.pdf`, and open the file to check it.
#
# Extra: add a third panel with the spectrum before and after a 60 Hz notch
# (`sp.signal.welch`, from the SciPy stretch), or show the 0.5 block as well.

# %% tags=["solution"]
colours = {0.2: "tab:red", 0.8: "tab:blue"}
names = {0.2: "right blocks (80% right)", 0.8: "left blocks (80% left)"}

fig, axd = plt.subplot_mosaic([["A", "B"]], figsize=(11, 4), layout="constrained")

ax = axd["A"]
for block in [0.2, 0.8]:
    ax.plot(grid, p_right(grid, block), color=colours[block], label=names[block])
    ax.plot(psychometric.index, psychometric[block], "o", color=colours[block])
ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("fraction of rightward choices")
ax.legend(frameon=False, loc="upper left")
ax.set_title("A", loc="left", fontweight="bold")

ax = axd["B"]
ax.plot(time, raw, color="lightgrey", linewidth=1, label="raw")
ax.plot(time, slow, color="black", linewidth=1, label="1-30 Hz")
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (µV)")
ax.legend(frameon=False, loc="upper right")
ax.set_title("B", loc="left", fontweight="bold")

for ax in axd.values():
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("capstone.pdf")
