# %% [markdown]
# # Day 2 · pandas
#
# **Block:** pandas (60 min, about 12 of them slides)
#
# We stay with yesterday's recording, and now look at its **tables**, which
# pandas is made for:
#
# * **trials**: one row per trial of the task: when the stimulus appeared, its
#   contrast (negative on the left, positive on the right, in %), the block's
#   prior (`probability_left`), the mouse's choice (-1 left, 1 right), whether
#   it was correct, and its response time.
# * **units**: one row per neuron (unit) found by spike sorting: the channel it
#   was largest on, that channel's brain area and height above the probe tip,
#   the sorter's quality label, and a few measurements.
#
# 1. **Reading tabular data** (10 min): Reading Other Data, Inspecting Data, Writing Data
# 2. **Selecting data** (15 min): Selection of Individual Values, Extent of
#    Slicing, Selecting Indices, Practice with Selection, Boolean masks
# 3. **Group by: split-apply-combine** (15 min): yesterday's psychometric curve
#    in one line, then a grouping question of your own
# 4. **Tidy data** (8 min): wide tables to long ones, ready for seaborn
# 5. **Stretch:** Many Ways of Access, then merging
#
# Sections 1 to 3 follow the exercises of Software Carpentry's [Plotting and
# Programming in Python](https://swcarpentry.github.io/python-novice-gapminder/),
# episodes [Reading Tabular Data into
# DataFrames](https://swcarpentry.github.io/python-novice-gapminder/07-reading-tabular.html)
# and [Pandas
# DataFrames](https://swcarpentry.github.io/python-novice-gapminder/08-data-frames.html)
# (CC BY 4.0), which use the gapminder data from the slides.

# %%
import h5py
import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. Reading tabular data
#
# A CSV file goes straight into a DataFrame with `pd.read_csv`. Our tables live
# in the HDF5 file instead, one array per column, so this small function reads
# one group of the file into a DataFrame. Text is stored as bytes in HDF5, so it
# converts those columns to strings.

# %%
def read_table(group):
    """Read one group of ../data/ibl_session.h5 into a DataFrame."""
    with h5py.File("../data/ibl_session.h5") as f:
        columns = {}
        for name, values in f[group].items():
            values = values[:]
            if values.dtype.kind == "S":
                values = values.astype(str)
            columns[name] = values
    return pd.DataFrame(columns)


trials = read_table("trials")
trials.index.name = "trial"
trials

# %%
trials.info()

# %% [markdown]
# ### Reading Other Data
#
# Read the `"units"` group into a variable called `units`. Make its `unit`
# column the row labels (the index) with `.set_index`, then display its summary
# statistics.

# %% tags=["solution"]
units = read_table("units").set_index("unit")
units.describe()

# %% [markdown]
# ### Inspecting Data
#
# Use `help(units.head)` and `help(units.tail)` to find out what
# `DataFrame.head` and `DataFrame.tail` do.
#
# 1. What method call will display the first three rows of this data?
# 2. What method call will display the last three **columns** of this data?
#    (Hint: you may need to change your view of the data.)

# %% tags=["solution"]
# 1.
units.head(n=3)

# %% tags=["solution"]
# 2. Transpose so the columns become rows, take the last three, transpose back.
units.T.tail(n=3).T

# %% [markdown] tags=["solution"]
# `units.iloc[:, -3:]` does the same thing more directly, and keeps each
# column's dtype (the transposes turn everything into `object`). You will meet
# `iloc` in the next section.

# %% [markdown]
# ### Writing Data
#
# As well as `read_csv`, pandas provides `to_csv` to write DataFrames to files.
# Write `units` to a file called `units.csv`, so a colleague without HDF5 could
# open it in a spreadsheet. Use `help` to find out how `to_csv` works, then read
# the file back in with `pd.read_csv` to check it. Which argument makes `unit`
# the index again?

# %% tags=["solution"]
units.to_csv("units.csv")
pd.read_csv("units.csv", index_col="unit").head(3)

# %% [markdown] tags=["solution"]
# `index_col="unit"`. `help(pd.to_csv)` fails, because `to_csv` is a **method**
# of a DataFrame, not a pandas function. Use `help(units.to_csv)` or
# `help(pd.DataFrame.to_csv)`.

# %% [markdown]
# ## 2. Selecting data
#
# * `df.iloc[row, col]` selects by **position** (0, 1, 2, ...)
# * `df.loc[row, col]` selects by **label** (unit `32`, column `"area"`)
# * `:` on its own means all rows or all columns
#
# Here the row labels of both tables are numbers, which makes the difference
# between the two easy to miss.

# %%
units = read_table("units").set_index("unit")
units.head()

# %% [markdown]
# ### Selection of Individual Values
#
# Write an expression to find the firing rate of unit 32.

# %% tags=["solution"]
units.loc[32, "firing_rate_Hz"]

# %% [markdown]
# ### Extent of Slicing
#
# 1. Do the two statements below produce the same output?
# 2. Based on this, what rule decides what is included (or not) in numerical
#    slices and in named slices in pandas?

# %%
print(units.iloc[0:2, 0:2])
print(units.loc[0:2, "channel":"area"])

# %% [markdown] tags=["answer"]
# No. The second gives an extra row (unit 2). A position slice `0:2` **excludes**
# the end, as everywhere else in Python. A label slice includes the end label,
# both for rows (`0:2` means the labels 0 to 2) and for columns
# (`"channel":"area"`).

# %% [markdown]
# ### Selecting Indices
#
# Explain in simple terms what `idxmin` and `idxmax` do in the cell below. When
# would you use these methods?

# %%
measurements = units[["firing_rate_Hz", "amplitude_uV", "presence_ratio"]]
print(measurements.idxmin())
print(measurements.idxmax())

# %% [markdown] tags=["answer"]
# For each column, `idxmin` returns the **row label** (here, the unit number)
# where the minimum value is, and `idxmax` does the same for the maximum. Use them
# when you want to know *which* row holds the extreme value, not the value
# itself: for example, which unit fires fastest.

# %% [markdown]
# ### Practice with Selection
#
# Using `trials`, write an expression to select each of the following:
#
# 1. The response time of every trial.
# 2. Everything about trial 89.
# 3. The columns from `contrast` to `choice`, for trials 500 onwards.
# 4. Each trial's response time as a multiple of the median response time.

# %% tags=["solution"]
# 1.
trials["response_time_s"]

# %% tags=["solution"]
# 2.
trials.loc[89, :]

# %% tags=["solution"]
# 3. A label slice: 500 to the end, contrast to choice inclusive.
trials.loc[500:, "contrast":"choice"]

# %% tags=["solution"]
# 4.
trials["response_time_s"] / trials["response_time_s"].median()

# %% [markdown]
# ### Boolean masks
#
# Comparisons give a Series of `True`/`False` that selects rows, exactly like
# NumPy masks. They work on text columns too. Here are the units on sites
# **above** the brain, which should not exist:

# %%
on_void = units["area"] == "void"
on_void.sum()

# %% [markdown]
# Using masks on `units`, make:
#
# 1. `n_slow`: the number of units firing below 1 spike per second
# 2. `good_ss`: the `firing_rate_Hz` and `amplitude_uV` columns, for units in
#    **`"SSs"`** that the sorter labelled `"good"`
# 3. `suspicious`: the unit numbers of the units that are either on `"void"`
#    sites **or** have a spike width of 0 or less (a waveform the
#    measurement could not handle). How many are there? (Hint: `.index`, and
#    brackets around each comparison.)

# %% tags=["solution"]
n_slow = (units["firing_rate_Hz"] < 1).sum()
is_good_ss = (units["area"] == "SSs") & (units["label"] == "good")
good_ss = units.loc[is_good_ss, ["firing_rate_Hz", "amplitude_uV"]]
suspicious = units.index[(units["area"] == "void") | (units["spike_width_ms"] <= 0)]
print(n_slow)
print(good_ss)
print(len(suspicious), suspicious[:10])

# %% [markdown]
# ## 3. Group by: split-apply-combine
#
# Does the mouse do the task? For every contrast, what fraction of the time did
# it turn the stimulus **right**? First add a column of `True`/`False`. Then
# **split** the trials by contrast, **apply** a mean to each group, and
# **combine** the results:

# %%
trials["chose_right"] = trials["choice"] == 1
psychometric = trials.groupby("contrast")["chose_right"].mean()
psychometric

# %% [markdown]
# **Talk it through with your neighbour:**
#
# * What does the mean of a column of `True`/`False` give?
# * Does the curve look like a mouse that can see the stimulus? What happens at
#   contrast 0, where there is nothing to see?
#
# The task has **blocks**: for a while the stimulus appears on the left 80% of
# the time (`probability_left` 0.8), then on the right 80% of the time (0.2).
# Group by two columns, and `unstack` one of them into columns:

# %%
by_block = trials.groupby(["probability_left", "contrast"])["chose_right"].mean()
by_block.unstack("probability_left").round(2)

# %% [markdown]
# Compare the 0.2 and 0.8 columns at contrast 0. Has the mouse learned the
# blocks?
#
# Yesterday you computed the same curve with a mask and broadcasting. Here is
# that NumPy version again, for the right blocks:

# %%
contrast = trials["contrast"].to_numpy()
y = trials["chose_right"].to_numpy()
right_block = trials["probability_left"].to_numpy() == 0.2
levels = np.unique(contrast)
at_level = contrast[:, np.newaxis] == levels
frac_right_rb = (at_level[right_block] & y[right_block, np.newaxis]).sum(axis=0) / at_level[right_block].sum(axis=0)

# %% [markdown]
# Check that pandas agrees: select the 0.2 column of the unstacked table and
# compare it with `frac_right_rb` using `np.allclose`. Which version would you
# rather write, and which would you rather read in six months?

# %% tags=["solution"]
np.allclose(by_block.unstack("probability_left")[0.2], frac_right_rb)

# %% [markdown]
# ### Your own grouping question
#
# Ask and answer **one** question of your own that needs `groupby`, about
# `trials` or `units`. For example:
#
# * Does the mouse respond faster when the stimulus is stronger? (Hint: group
#   the median response time by `trials["contrast"].abs()`. A Series can be the
#   thing you group by.)
# * Is the mouse more often correct in some blocks than in others?
# * How many units does each brain area have, how fast do they fire, and how
#   many are labelled `"good"`? (Hint: `.agg` with several functions.)

# %% tags=["solution"]
# Median response time against stimulus strength.
trials.groupby(trials["contrast"].abs())["response_time_s"].median().round(3)

# %% tags=["solution"]
# Fraction correct in each block.
trials.groupby("probability_left")["correct"].mean().round(3)

# %% tags=["solution"]
# Units per brain area.
units.groupby("area").agg(
    n_units=("firing_rate_Hz", "size"),
    median_rate_Hz=("firing_rate_Hz", "median"),
    n_good=("label", lambda label: (label == "good").sum()),
)

# %% [markdown] tags=["solution"]
# Responses get faster as the contrast rises, from about 0.5 s at 0% to under
# 0.3 s at 100%: a stronger stimulus is an easier decision. The mouse does worst
# in the unbiased (0.5) blocks, the first trials of the session, before it can
# use the prior. Half of the units are in the two somatosensory areas, and only
# about one in ten passes the sorter's quality checks.

# %% [markdown]
# ## 4. Tidy data
#
# seaborn, which you meet after the break, wants **tidy** ("long") tables: one
# row per observation and one column per variable. `trials` and `units` are
# already tidy. The unstacked psychometric curve is **wide**: one column per
# block, so the block is hidden in the column names instead of being a
# variable.

# %%
psychometric_wide = by_block.unstack("probability_left")
psychometric_wide

# %% [markdown]
# 1. Make `psychometric_long`, with the columns `contrast`, `probability_left`
#    and `fraction_right`, one row per block and contrast (27 rows). (Hint:
#    `reset_index()` turns the `contrast` index into a column, and `melt` with
#    `id_vars="contrast"` does the rest. Look up `var_name` and `value_name`.)

# %% tags=["solution"]
psychometric_long = psychometric_wide.reset_index().melt(
    id_vars="contrast", var_name="probability_left", value_name="fraction_right"
)
psychometric_long.head()

# %% [markdown]
# The spike counts make a wide table too. Averaged over trials, they give one
# row per unit and **one column per time bin**:

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
rates.iloc[:3, :6]

# %% [markdown]
# 2. Make `rates_long`, with the columns `unit`, `time_s` and `rate_Hz`: one row
#    per unit **and** time bin. What are the shapes of `rates` and `rates_long`,
#    and why?

# %% tags=["solution"]
rates_long = rates.reset_index().melt(id_vars="unit", var_name="time_s", value_name="rate_Hz")
print(rates.shape, rates_long.shape)
rates_long.head()

# %% [markdown] tags=["answer"]
# `rates` is (51, 30): 51 units by 30 bins. `rates_long` is (1530, 3): one row
# for each of the 51 × 30 unit-bin pairs, with the unit, the time and the rate
# as columns. Long tables are longer, but every variable is a column you can
# group by, filter on, or hand to seaborn.

# %% [markdown]
# ## 5. Stretch
#
# Work through any of these, in any order.

# %% [markdown]
# ### Many Ways of Access (from the same Software Carpentry episode)
#
# There are many ways to get at the same data: by label or by position, and as
# a `DataFrame` or as a `Series`. Suggest at least two different ways of doing each
# of the following on `units`:
#
# 1. Access a single column
# 2. Access a single row
# 3. Access an individual element
# 4. Access several columns
# 5. Access several rows
# 6. Access a subset of specific rows and columns
# 7. Access a subset of row and column ranges
#
# Check the type of each result with `type(...)`. When do you get a `Series`, and
# when a `DataFrame`?

# %% tags=["solution"]
# 1. Single column
units["area"]                       # Series
units[["area"]]                     # DataFrame
units.iloc[:, 1]                    # Series, by position

# 2. Single row
units.loc[32]                       # Series
units.loc[[32]]                     # DataFrame
units.iloc[32]                      # Series, by position (unit 32 is row 32)

# 3. Single element
units.loc[32, "firing_rate_Hz"]
units.iloc[32, 4]
units.at[32, "firing_rate_Hz"]      # fast scalar access

# 4. Several columns
units[["area", "firing_rate_Hz"]]
units.iloc[:, [1, 4]]

# 5. Several rows
units.loc[[24, 32]]
units.iloc[[24, 32]]

# 6. Specific rows and columns
units.loc[[24, 32], ["area", "firing_rate_Hz"]]
units.iloc[[24, 32], [1, 4]]

# 7. Ranges
units.loc[30:33, "channel":"depth_um"]
units.iloc[30:34, 0:3]   # rows 30 to 33, the same as the .loc line: the end is excluded

# %% [markdown] tags=["solution"]
# A single label or position gives a `Series`. A **list** of labels (even a list
# of one, `[["area"]]`) or a slice keeps the result two-dimensional, so it is a
# `DataFrame`. Here unit numbers and positions happen to be equal, because the
# units are numbered 0, 1, 2, ... in order. After filtering or sorting they no
# longer would be, and `loc` and `iloc` would give different rows.

# %% [markdown]
# ### Merging: which layer is each unit in?
#
# The `"electrodes"` group has one row per channel, including its cortical
# `layer`. Read it into a DataFrame, then use `pd.merge` (or `units.merge`) on
# the `channel` column to give every unit the layer of its channel. How many
# units are in each layer? (Sites outside the cortex, in `PIR` or above the
# brain, have no layer: an empty string.)

# %% tags=["solution"]
electrodes = read_table("electrodes")
units_with_layer = units.reset_index().merge(electrodes[["channel", "layer"]], on="channel")
units_with_layer.groupby("layer")["unit"].count()

# %% [markdown]
# ### Further reading
#
# * Python Data Science Handbook: [Handling Missing
#   Data](https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html),
#   [Combining Datasets: Merge and
#   Join](https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html),
#   [Hierarchical
#   Indexing](https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html)
#   and [Working with Time
#   Series](https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html).
#   The website is the first edition, so where it uses `DataFrame.append` (removed
#   in pandas 2.0), use `pd.concat` instead.
# * The [pandas_exercises](https://github.com/guipsamora/pandas_exercises) repository.
