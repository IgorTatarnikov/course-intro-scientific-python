# %% [markdown]
# # Day 2 · scikit-learn
#
# **Block:** scikit-learn (30 min, including slides and live coding)
#
# 1. **Fit a model** (10 min): predict the mouse's choice from the contrast and
#    the block, and check it on trials it has not seen
# 2. **Look inside the model** (10 min): its coefficients, and its curves on top
#    of the psychometric curve from the pandas block
# 3. **Stretch:** a better feature, whether the block helps, cross-validation,
#    and decoding the choice from the neurons
#
# Every scikit-learn model works the same way: make it, `fit` it on `X` and
# `y`, then `predict` or `score`. Learn that pattern once and it carries over
# to the other models. The [user guide](https://scikit-learn.org/stable/user_guide.html)
# covers each model in depth; [Scientific Python Lectures, scikit-learn: machine
# learning in Python](https://lectures.scientific-python.org/packages/scikit-learn/index.html)
# is a longer introduction.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(seed=0)


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
    spike_counts = f["spike_counts/data"][:]

# %% [markdown]
# A model wants its data as a table `X` with **one row per sample** (here, a
# trial) and **one column per feature**, plus `y`, the answer for each sample.
# A DataFrame works as `X`, and so does a 2-D NumPy array. Our features are the
# contrast and the block; the answer is whether the mouse chose right:

# %%
X = trials[["contrast", "probability_left"]]
y = trials["chose_right"]
X.shape, y.shape

# %% [markdown]
# And the psychometric curve from part 3 of the pandas block, one column per
# block, to compare the model with:

# %%
psychometric = trials.groupby(["contrast", "probability_left"])["chose_right"].mean().unstack()
psychometric.round(2)

# %% [markdown]
# ## 1. Fit a model
#
# **Logistic regression** predicts the probability of a yes/no answer (here: did
# the mouse choose right?). Each feature gets a weight. The weighted sum of the
# features is squashed through an S-shaped curve into a probability between 0
# and 1. Despite its name, it is used for classification.
#
# 1. Split `X` and `y` into a training set and a test set with
#    `train_test_split(X, y, test_size=0.25, random_state=0)`. It returns four
#    tables: `X_train, X_test, y_train, y_test`. How many trials are in each?
# 2. Make a `LogisticRegression()` called `model` and `fit` it on the
#    **training** set.
# 3. `model.score(X, y)` gives the fraction of trials whose choice the model
#    predicts correctly. Score it on the training set and on the test set.
# 4. Is that good? Work out the **baseline**: the score you would get by always
#    guessing the more common choice in the test set. (Hint: the mean of a
#    boolean column is the fraction that is `True`.)

# %% tags=["solution"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(len(y_train), len(y_test))

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"train {model.score(X_train, y_train):.3f}, test {model.score(X_test, y_test):.3f}")

baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"baseline {baseline:.3f}")

# %% [markdown] tags=["answer"]
# 399 training trials and 134 test trials. The model gets about 85% of the test
# trials right, against 53% for always guessing the more common choice, and
# about the same on the training set, so it is not just memorising the trials it
# was fitted on. Always compare a score with a baseline: 85% sounds good on its
# own, but so would 53% to someone who did not know the classes were balanced.

# %% [markdown]
# ## 2. Look inside the model
#
# 1. `model.coef_` holds one weight per feature (in the order of the columns of
#    `X`) and `model.intercept_` the constant. When a feature goes up, does the
#    predicted probability of choosing right go up or down?
# 2. Make `grid`, 201 contrasts from -100 to 100. For each kind of block, make a
#    DataFrame with the same columns as `X`: the grid as `contrast`, and the
#    block's `probability_left` (0.2 or 0.8) in every row. (Hint:
#    `pd.DataFrame({"contrast": grid, "probability_left": 0.2})` repeats the
#    single value.)
# 3. `model.predict_proba(...)` returns one column per class, in the order of
#    `model.classes_`. Take the probability of `True` (chose right) for each
#    block.
# 4. Plot the two model curves against `grid`, with the data
#    (`psychometric[0.2]` and `psychometric[0.8]` against `psychometric.index`)
#    as points in the same colours. Where does the model miss the data?

# %% tags=["solution"]
print(model.coef_, model.intercept_)

# %% [markdown] tags=["answer"]
# The contrast weight is positive, so the predicted probability of a right
# choice rises with contrast. The `probability_left` weight is negative, so it
# falls in left blocks: the model has picked up the difference between the
# block curves. Read the **signs**, not the sizes: `LogisticRegression`
# regularises by default (its `C` parameter), which shrinks the weights, and
# features on different scales get weights on different scales.

# %% tags=["solution"]
grid = np.linspace(-100, 100, 201)
grid_rb = pd.DataFrame({"contrast": grid, "probability_left": 0.2})
grid_lb = pd.DataFrame({"contrast": grid, "probability_left": 0.8})
print(model.classes_)  # column 1 is True
p_right_rb = model.predict_proba(grid_rb)[:, 1]
p_right_lb = model.predict_proba(grid_lb)[:, 1]

fig, ax = plt.subplots()
ax.plot(grid, p_right_rb, color="tab:red", label="model, right blocks")
ax.plot(grid, p_right_lb, color="tab:blue", label="model, left blocks")
ax.plot(psychometric.index, psychometric[0.2], "o", color="tab:red", label="data, right blocks")
ax.plot(psychometric.index, psychometric[0.8], "o", color="tab:blue", label="data, left blocks")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("probability of choosing right")
ax.legend();

# %% [markdown] tags=["answer"]
# The model's curves rise far too slowly. The data are already near 1 at 12.5%
# and 25% contrast, but the model says only 0.73 and 0.86 in right blocks, and
# 0.46 and 0.66 in left blocks. Logistic regression assumes each extra percent
# of contrast changes the odds by the same factor, all the way to 100%, and
# these data level off well before that. Plotting a model over its data is the
# quickest way to see a misfit that a single score hides. The model is only as
# good as the features you give it (stretch: a better feature).

# %% [markdown]
# ## 3. Stretch
#
# ### A better feature
#
# Make `X_tanh`, a DataFrame with a `tanh_contrast` column,
# `np.tanh(contrast / 25)`, which rises steeply near 0 and flattens out by
# about ±50%, and the `probability_left` column. Refit on the same split (the
# same `random_state`), score it on the test set, and add its curves to the
# plot. Is it better?

# %% tags=["solution"]
def tanh_features(contrast, probability_left):
    return pd.DataFrame({"tanh_contrast": np.tanh(contrast / 25), "probability_left": probability_left})


X_tanh = tanh_features(trials["contrast"], trials["probability_left"])
Xt_train, Xt_test, _, _ = train_test_split(X_tanh, y, test_size=0.25, random_state=0)
model_tanh = LogisticRegression().fit(Xt_train, y_train)
print(f"test {model_tanh.score(Xt_test, y_test):.3f}")

fig, ax = plt.subplots()
ax.plot(grid, model_tanh.predict_proba(tanh_features(grid, 0.2))[:, 1], color="tab:red", label="tanh model, right blocks")
ax.plot(grid, model_tanh.predict_proba(tanh_features(grid, 0.8))[:, 1], color="tab:blue", label="tanh model, left blocks")
ax.plot(psychometric.index, psychometric[0.2], "o", color="tab:red")
ax.plot(psychometric.index, psychometric[0.8], "o", color="tab:blue")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("probability of choosing right")
ax.legend();

# %% [markdown] tags=["solution"]
# The curves now follow the data closely. The test score barely moves (0.851 to
# 0.858), because `score` only counts which side is more likely, and both
# models agree on that for most trials. A score can hide a large difference in
# the predicted probabilities, so look at a plot, or at a metric that uses
# them (`sklearn.metrics.log_loss`).
#
# ### Does the block help? One split is not enough
#
# 1. Fit a model on the contrast alone (`X_train[["contrast"]]`: the double
#    brackets keep it a DataFrame, so it stays 2-D) and score it on the test
#    set. Does adding the block help?
# 2. One random split is one roll of the dice. `cross_val_score(model, X, y,
#    cv=5)` splits the data five ways, fits and scores on each, and returns the
#    five scores. Compare the two models on their mean cross-validated score.

# %% tags=["solution"]
model_contrast = LogisticRegression().fit(X_train[["contrast"]], y_train)
print(f"contrast only, test {model_contrast.score(X_test[['contrast']], y_test):.3f}")

print(f"both, cross-validated          {cross_val_score(LogisticRegression(), X, y, cv=5).mean():.3f}")
print(f"contrast only, cross-validated {cross_val_score(LogisticRegression(), X[['contrast']], y, cv=5).mean():.3f}")

# %% [markdown] tags=["solution"]
# On our one split the block seems to help (0.836 to 0.851). Cross-validated,
# the two are the same (about 0.855). Small differences between models on one
# split are often luck, so compare models with cross-validation.
#
# ### Decode the choice from the neurons
#
# Can you predict the mouse's choice from what its neurons did? Use the total
# spike count of each of the 51 units after the stimulus as the features:
# `X_spikes = spike_counts[:, :, 10:].sum(axis=2).T`, a NumPy array of shape
# (533, 51).
#
# 1. The units fire at very different rates, so scale each feature to mean 0
#    and standard deviation 1 first. `make_pipeline(StandardScaler(),
#    LogisticRegression(max_iter=1000))` chains the two steps into one model.
#    Get its 5-fold cross-validated scores.
# 2. Two controls: the same with `y` shuffled (`rng.permutation(y)`), and the
#    same with the spike counts from **before** the stimulus (bins 0 to 9).
#    What do they tell you?

# %% tags=["solution"]
X_spikes = spike_counts[:, :, 10:].sum(axis=2).T
decoder = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(decoder, X_spikes, y, cv=5)
print("after stimulus:", scores.round(2), scores.mean().round(3))
print("shuffled choices:", cross_val_score(decoder, X_spikes, rng.permutation(y), cv=5).mean().round(3))
X_before = spike_counts[:, :, :10].sum(axis=2).T
print("before stimulus:", cross_val_score(decoder, X_before, y, cv=5).mean().round(3))

# %% [markdown] tags=["solution"]
# The decoder scores about 0.68, against about 0.50 for both controls: the
# controls show what "no information" scores, so the 0.68 means something. The
# five folds range from about 0.52 to 0.77, so report the spread as well as the
# mean. For a classifier, `cv=5` uses `StratifiedKFold` without shuffling,
# which keeps the trials roughly in session order, so each fold tests on a
# different part of the session; that is one reason the folds differ.
