# %% [markdown]
# # Day 2 · seaborn
#
# **Block:** seaborn (15 min)
#
# The first part is a **live demo**: follow along, run the cells and change
# things. The demo mirrors the Python Data Science Handbook's [Visualization
# with Seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html),
# but uses the gapminder data from Day 1. Then make **one plot of your
# own**.
#
# The [seaborn tutorial](https://seaborn.pydata.org/tutorial.html) and
# [example gallery](https://seaborn.pydata.org/examples/index.html) are the best
# places to look things up.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Tidy ("long") data
#
# seaborn wants **one row per observation** and **one column per variable**. The
# gapminder file is "wide": one column per measurement *and* year. `pd.wide_to_long`
# splits the column names at the underscore:

# %%
wide = pd.read_csv("../data/gapminder_all.csv")
gapminder = pd.wide_to_long(
    wide, stubnames=["gdpPercap", "lifeExp", "pop"], i="country", j="year", sep="_"
).reset_index()
gapminder.head()

# %%
gapminder.shape, wide.shape

# %% [markdown]
# ## Demo
#
# ### Relationships: `relplot`
#
# Each argument maps a **column** to a visual property. The legend is built for us.

# %%
g = sns.relplot(
    data=gapminder[gapminder["year"] == 2007],
    x="gdpPercap",
    y="lifeExp",
    hue="continent",
    size="pop",
    sizes=(10, 800),
    alpha=0.7,
    height=5,
    aspect=1.4,
)
g.set(xscale="log", xlabel="GDP per capita (log scale)", ylabel="Life expectancy (years)");

# %% [markdown]
# The same plot in plain Matplotlib needs a loop over continents, manual colours
# and a hand-made size legend. That is the main reason to reach for seaborn when
# your data is already in a DataFrame.
#
# ### Change over time, with uncertainty: `lineplot`
#
# Several countries share each (year, continent) pair. seaborn **aggregates** them
# (mean by default) and shades a 95% confidence interval:

# %%
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=gapminder, x="year", y="lifeExp", hue="continent", ax=ax);

# %% [markdown]
# ### Distributions: `histplot`, `kdeplot`, `boxplot`, `violinplot`

# %% [markdown]
# Watch out: `year` is a number, so as a `hue` seaborn would give it a
# *continuous* colour scale, which makes two years hard to tell apart. Converting
# it to a string makes it a category with distinct colours.

# %%
first_last = gapminder[gapminder["year"].isin([1952, 2007])].astype({"year": str})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(
    data=first_last,
    x="lifeExp",
    hue="year",
    fill=True,
    ax=axes[0],
)
sns.boxplot(
    data=gapminder[gapminder["year"] == 2007],
    x="continent",
    y="gdpPercap",
    log_scale=True,
    ax=axes[1],
);

# %% [markdown]
# ### Small multiples: one panel per category with `col=`

# %%
sns.displot(
    data=gapminder[gapminder["year"] == 2007],
    x="lifeExp",
    col="continent",
    col_wrap=3,
    height=2.5,
    bins=10,
);

# %% [markdown]
# ### Regression: `lmplot`
#
# A straight-line fit with its confidence band, here against log GDP:

# %%
gapminder["logGdp"] = np.log10(gapminder["gdpPercap"])
sns.lmplot(data=gapminder[gapminder["year"] == 2007], x="logGdp", y="lifeExp", height=4, aspect=1.4);

# %% [markdown]
# Every seaborn figure is still Matplotlib underneath: axes-level functions
# (`lineplot`, `boxplot`) take `ax=` and return an `Axes`; figure-level
# functions (`relplot`, `displot`, `lmplot`, `catplot`) return a `FacetGrid` `g`,
# with `g.figure` and `g.axes`. Customise and save it with the Matplotlib from
# the previous block.
#
# ## Your turn: one plot of your own
#
# Use `gapminder` to make one seaborn plot that answers a question **you** find
# interesting. Give it proper axis labels and save it to a file. Ideas:
#
# * How has the *spread* of life expectancy within each continent changed between
#   1952 and 2007? (`catplot` with `kind="box"` or `kind="violin"`)
# * Which countries' populations grew fastest? (compute a growth column first)
# * Does the GDP versus life expectancy relationship look the same in 1952 as in
#   2007? (`relplot` with `col="year"`)

# %% tags=["solution"]
# One possible answer: the spread of life expectancy within each continent, 1952 and 2007.
g = sns.catplot(
    data=first_last,
    x="continent",
    y="lifeExp",
    hue="year",
    kind="violin",
    split=True,
    inner="quart",
    density_norm="width",  # every violin equally wide; Oceania has only two countries
    cut=0,  # do not draw the density beyond the observed values
    height=4,
    aspect=2,
)
g.set_axis_labels("", "Life expectancy (years)")
g.figure.suptitle("Life expectancy rose everywhere; Africa's spread widened", y=1.03)
g.savefig("life_expectancy_by_continent.png", dpi=150)
