# %% [markdown]
# # Day 1 · pandas
#
# **Block:** pandas (55 min, about 10 of them slides)
#
# We use the **gapminder** data here and again for plotting on Day 2: GDP per capita (and, in
# `gapminder_all.csv`, life expectancy and population) for 142 countries from
# 1952 to 2007, one CSV file per continent.
#
# 1. **Reading tabular data** (10 min): Reading Other Data, Inspecting Data, Writing Data
# 2. **Selecting data** (15 min): Selection of Individual Values, Extent of
#    Slicing, Selecting Indices, Practice with Selection, Boolean masks
# 3. **Group by: split-apply-combine** (15 min), finishing with a grouping question of your own
# 4. **Stretch:** Many Ways of Access, then missing data, merging and reshaping
#
# Sections 1 to 3 are adapted from Software Carpentry's [Plotting and Programming
# in Python](https://swcarpentry.github.io/python-novice-gapminder/), episodes
# [Reading Tabular Data into
# DataFrames](https://swcarpentry.github.io/python-novice-gapminder/07-reading-tabular.html)
# and [Pandas
# DataFrames](https://swcarpentry.github.io/python-novice-gapminder/08-data-frames.html)
# (CC BY 4.0).

# %%
import pandas as pd

# %% [markdown]
# ## 1. Reading tabular data
#
# `read_csv` turns a CSV file into a `DataFrame`. `index_col` picks the column
# whose values become the row labels.

# %%
data_oceania = pd.read_csv("../data/gapminder_gdp_oceania.csv", index_col="country")
data_oceania

# %%
data_oceania.info()

# %% [markdown]
# ### Reading Other Data
#
# Read the data in `gapminder_gdp_americas.csv` (in the same `../data/` folder)
# into a variable called `data_americas` and display its summary statistics.

# %% tags=["solution"]
data_americas = pd.read_csv("../data/gapminder_gdp_americas.csv", index_col="country")
data_americas.describe()

# %% [markdown]
# ### Inspecting Data
#
# Use `help(data_americas.head)` and `help(data_americas.tail)` to find out what
# `DataFrame.head` and `DataFrame.tail` do.
#
# 1. What method call will display the first three rows of this data?
# 2. What method call will display the last three **columns** of this data?
#    (Hint: you may need to change your view of the data.)

# %% tags=["solution"]
# 1.
data_americas.head(n=3)

# %% tags=["solution"]
# 2. Transpose so the columns become rows, take the last three, transpose back.
data_americas.T.tail(n=3).T

# %% [markdown] tags=["solution"]
# `data_americas.iloc[:, -3:]` does the same thing more directly. You will meet
# `iloc` in the next section.

# %% [markdown]
# ### Writing Data
#
# As well as `read_csv`, pandas provides `to_csv` to write DataFrames to files.
# Write one of your DataFrames to a file called `processed.csv`. Use `help` to
# find out how `to_csv` works, then read the file back in to check it.

# %% tags=["solution"]
data_americas.to_csv("processed.csv")
pd.read_csv("processed.csv", index_col="country").head(3)

# %% [markdown] tags=["solution"]
# `help(pd.to_csv)` fails, because `to_csv` is a **method** of a DataFrame, not
# a pandas function. Use `help(data_americas.to_csv)` or `help(pd.DataFrame.to_csv)`.

# %% [markdown]
# ## 2. Selecting data
#
# * `df.iloc[row, col]` selects by **position** (0, 1, 2, ...)
# * `df.loc[row, col]` selects by **label** (`"Albania"`, `"gdpPercap_1952"`)
# * `:` on its own means all rows or all columns

# %%
data_europe = pd.read_csv("../data/gapminder_gdp_europe.csv", index_col="country")
data_europe.head()

# %% [markdown]
# ### Selection of Individual Values
#
# Write an expression to find the GDP per capita of Serbia in 2007.

# %% tags=["solution"]
data_europe.loc["Serbia", "gdpPercap_2007"]

# %% [markdown]
# ### Extent of Slicing
#
# 1. Do the two statements below produce the same output?
# 2. Based on this, what rule decides what is included (or not) in numerical
#    slices and in named slices in pandas?

# %%
print(data_europe.iloc[0:2, 0:2])
print(data_europe.loc["Albania":"Belgium", "gdpPercap_1952":"gdpPercap_1962"])

# %% [markdown] tags=["answer"]
# No. The second gives an extra row (Belgium) and an extra column (1962).
# A numerical slice `0:2` **excludes** the end, as everywhere else in Python. A
# named slice `"Albania":"Belgium"` **includes** the end label.

# %% [markdown]
# ### Selecting Indices
#
# Explain in simple terms what `idxmin` and `idxmax` do in the cell below. When
# would you use these methods?

# %%
print(data_europe.idxmin())
print(data_europe.idxmax())

# %% [markdown] tags=["answer"]
# For each column, `idxmin` returns the **row label** (here, the country) where
# the minimum value is, and `idxmax` does the same for the maximum. Use them when
# you want to know *which* row holds the extreme value, not the value itself.

# %% [markdown]
# ### Practice with Selection
#
# Using `data_europe`, write an expression to select each of the following:
#
# 1. GDP per capita for all countries in 1982.
# 2. GDP per capita for Denmark for all years.
# 3. GDP per capita for all countries for years *after* 1985.
# 4. GDP per capita for each country in 2007 as a multiple of GDP per capita for
#    that country in 1952.

# %% tags=["solution"]
# 1.
data_europe["gdpPercap_1982"]

# %% tags=["solution"]
# 2.
data_europe.loc["Denmark", :]

# %% tags=["solution"]
# 3. No column is called gdpPercap_1985, but labels are sorted strings, so the
#    slice starts at the first label after it.
data_europe.loc[:, "gdpPercap_1985":]

# %% tags=["solution"]
# 4.
data_europe["gdpPercap_2007"] / data_europe["gdpPercap_1952"]

# %% [markdown]
# ### Boolean masks
#
# Comparisons give a DataFrame (or Series) of `True`/`False`, which can be used to
# select rows, exactly like NumPy masks:

# %%
rich_2007 = data_europe["gdpPercap_2007"] > 30000
data_europe.loc[rich_2007, ["gdpPercap_1952", "gdpPercap_2007"]]

# %% [markdown]
# Using masks, make:
#
# 1. `n_poor_1952`: the number of countries with a GDP per capita below 2000
#    in 1952
# 2. `middle_1982`: the 1952 and 2007 columns, for countries whose GDP per
#    capita in 1982 was between 10000 and 20000
# 3. `fast_growers`: the names of the countries whose GDP per capita grew
#    more than five-fold from 1952 to 2007 (Hint: `.index`)

# %% tags=["solution"]
n_poor_1952 = (data_europe["gdpPercap_1952"] < 2000).sum()
in_range = (data_europe["gdpPercap_1982"] > 10000) & (data_europe["gdpPercap_1982"] < 20000)
middle_1982 = data_europe.loc[in_range, ["gdpPercap_1952", "gdpPercap_2007"]]
growth = data_europe["gdpPercap_2007"] / data_europe["gdpPercap_1952"]
fast_growers = data_europe.index[growth > 5]
print(n_poor_1952)
print(middle_1982)
print(fast_growers)

# %% [markdown]
# ## 3. Group by: split-apply-combine
#
# How do European countries split by wealth? First mark, for every country and
# year, whether its GDP was above the European average for that year. Then
# score each country by the fraction of years it was above average:

# %%
mask_higher = data_europe > data_europe.mean()
wealth_score = mask_higher.aggregate("sum", axis=1) / len(data_europe.columns)
wealth_score.sort_values()

# %% [markdown]
# **Talk it through with your neighbour:**
#
# * What shape is `data_europe.mean()`, and how does it line up against `data_europe`?
# * `True` counts as 1 and `False` as 0 when summed. What does `axis=1` do here?
#
# Now **split** the countries by score, **apply** a sum to each group and
# **combine** the results into one table:

# %%
data_europe.groupby(wealth_score).sum()

# %% [markdown]
# ### Your own grouping question
#
# `gapminder_all.csv` has every country, with a `continent` column and three
# measurements per year: `gdpPercap_*`, `lifeExp_*` and `pop_*`.
#
# Ask and answer **one** question of your own that needs `groupby`. For example:
#
# * Which continent had the highest median life expectancy in 1952, and in 2007?
# * How many countries does each continent have, and what is its total population in 2007?
# * Within each continent, which country had the highest GDP per capita in 2007?
#   (Hint: `idxmax`.)

# %%
data_all = pd.read_csv("../data/gapminder_all.csv", index_col="country")
data_all.iloc[:5, :4]

# %% tags=["solution"]
# Median life expectancy by continent, at the start and end of the data.
data_all.groupby("continent")[["lifeExp_1952", "lifeExp_2007"]].median()

# %% tags=["solution"]
# Number of countries and total population (in millions) per continent in 2007.
data_all.groupby("continent")["pop_2007"].agg(["count", "sum"]).assign(
    sum=lambda df: (df["sum"] / 1e6).round()
)

# %% tags=["solution"]
# Richest country per continent in 2007.
data_all.groupby("continent")["gdpPercap_2007"].idxmax()

# %% [markdown]
# ## 4. Stretch
#
# Work through any of these, in any order.

# %% [markdown]
# ### Many Ways of Access (from the same Software Carpentry episode)
#
# There are many ways to get at the same data: by label or by position, and as
# a `DataFrame` or as a `Series`. Suggest at least two different ways of doing each
# of the following on `data_europe`:
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
data_europe["gdpPercap_1952"]       # Series
data_europe[["gdpPercap_1952"]]     # DataFrame
data_europe.iloc[:, 0]              # Series, by position

# 2. Single row
data_europe.loc["Denmark"]          # Series
data_europe.loc[["Denmark"]]        # DataFrame
data_europe.iloc[7]                 # Series, by position (Denmark is row 7)

# 3. Single element
data_europe.loc["Denmark", "gdpPercap_1952"]
data_europe.iloc[7, 0]
data_europe.at["Denmark", "gdpPercap_1952"]  # fast scalar access

# 4. Several columns
data_europe[["gdpPercap_1952", "gdpPercap_2007"]]
data_europe.iloc[:, [0, -1]]

# 5. Several rows
data_europe.loc[["Denmark", "Norway"]]
data_europe.iloc[[7, 18]]

# 6. Specific rows and columns
data_europe.loc[["Denmark", "Norway"], ["gdpPercap_1952", "gdpPercap_2007"]]
data_europe.iloc[[7, 18], [0, -1]]

# 7. Ranges
data_europe.loc["Denmark":"Germany", "gdpPercap_1952":"gdpPercap_1962"]
data_europe.iloc[7:11, 0:3]   # rows 7 to 10, the same as the .loc line: the end is excluded

# %% [markdown] tags=["solution"]
# A single label or position gives a `Series`. A **list** of labels (even a list
# of one, `[["gdpPercap_1952"]]`) or a slice keeps the result two-dimensional,
# so it is a `DataFrame`.

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
#
# A small taste of reshaping, which you need for seaborn on Day 2. The
# **wide** table has one column per year; the **long** table has one row per
# country and year:

# %%
gdp_long = (
    data_europe.reset_index()
    .melt(id_vars="country", var_name="year", value_name="gdpPercap")
    .assign(year=lambda df: df["year"].str.removeprefix("gdpPercap_").astype(int))
)
gdp_long.head()
