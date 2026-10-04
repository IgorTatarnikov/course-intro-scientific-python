# %% [markdown]
# # Day 1 · scikit-learn
#
# **Block:** scikit-learn (35 min, including slides and live coding)
#
# 1. **Fit a model** (10 min): predict the mouse's choice from the contrast and
#    the block, and check it on trials it has not seen
# 2. **Look inside the model** (10 min): its coefficients, and its curves on top
#    of the psychometric curve from the NumPy block
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(seed=0)

with h5py.File("../data/ibl_session.h5") as f:
    contrast = f["trials/contrast"][:]
    probability_left = f["trials/probability_left"][:]
    choice = f["trials/choice"][:]
    spike_counts = f["spike_counts/data"][:]

# %% [markdown]
# These are the results of parts 2c and 3c of the NumPy block, so you can start
# here even if you did not finish them: the features `X` and the answers `y`,
# and the psychometric curve in each kind of block.

# %%
X = np.column_stack([contrast, probability_left])  # one row per trial
y = choice == 1  # True where the mouse chose right

levels = np.unique(contrast)
at_level = contrast[:, np.newaxis] == levels
right_block = probability_left == 0.2
left_block = probability_left == 0.8
frac_right_rb = (at_level[right_block] & y[right_block, np.newaxis]).sum(axis=0) / at_level[right_block].sum(axis=0)
frac_right_lb = (at_level[left_block] & y[left_block, np.newaxis]).sum(axis=0) / at_level[left_block].sum(axis=0)

X.shape, y.shape

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
#    arrays: `X_train, X_test, y_train, y_test`. How many trials are in each?
# 2. Make a `LogisticRegression()` called `model` and `fit` it on the
#    **training** set.
# 3. `model.score(X, y)` gives the fraction of trials whose choice the model
#    predicts correctly. Score it on the training set and on the test set.
# 4. Is that good? Work out the **baseline**: the score you would get by always
#    guessing the more common choice in the test set. (Hint: the mean of a
#    boolean array is the fraction that is `True`.)

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
#    `X`) and `model.intercept_` the constant. What does the **sign** of each
#    weight tell you about the mouse?
# 2. Make `grid`, 201 contrasts from -100 to 100. For each kind of block, build
#    a feature array for the grid, shape (201, 2), with the block's
#    `probability_left` (0.2 or 0.8) in every row of column 1. (Hint:
#    `np.full_like(grid, 0.2)`.)
# 3. `model.predict_proba(X_grid)` returns one column per class, in the order of
#    `model.classes_`. Take the probability of `True` (chose right) for each
#    block.
# 4. Plot the two model curves against `grid`, with the data from the NumPy
#    block (`frac_right_rb` and `frac_right_lb` against `levels`) as points on
#    top. Where does the model miss the data?

# %% tags=["solution"]
print(model.coef_, model.intercept_)

# %% [markdown] tags=["answer"]
# The contrast weight is positive: the more the stimulus is on the right, the
# more likely the mouse turns right. The `probability_left` weight is negative:
# in blocks where the stimulus is usually on the left, the mouse is less likely
# to choose right, whatever it sees. That is the prior you saw in the NumPy
# block, now as a number.

# %% tags=["solution"]
grid = np.linspace(-100, 100, 201)
X_grid_rb = np.column_stack([grid, np.full_like(grid, 0.2)])
X_grid_lb = np.column_stack([grid, np.full_like(grid, 0.8)])
print(model.classes_)  # column 1 is True
p_right_rb = model.predict_proba(X_grid_rb)[:, 1]
p_right_lb = model.predict_proba(X_grid_lb)[:, 1]

fig, ax = plt.subplots()
ax.plot(grid, p_right_rb, color="tab:red", label="model, right blocks")
ax.plot(grid, p_right_lb, color="tab:blue", label="model, left blocks")
ax.plot(levels, frac_right_rb, "o", color="tab:red", label="data, right blocks")
ax.plot(levels, frac_right_lb, "o", color="tab:blue", label="data, left blocks")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("probability of choosing right")
ax.legend();

# %% [markdown] tags=["answer"]
# The model's curves rise far too slowly. The mouse is already nearly always
# right at 12.5% and 25% contrast, but the model says only 0.73 and 0.86 in
# right blocks, and 0.46 and 0.66 in left blocks. Logistic regression assumes each extra percent of contrast changes the
# odds by the same factor, all the way to 100%. The mouse's vision saturates:
# going from 0 to 25% matters far more than going from 25 to 100%. The model
# is only as good as the features you give it (stretch: a better feature).

# %% [markdown]
# ## 3. Stretch
#
# ### A better feature
#
# Replace the contrast column by `np.tanh(contrast / 25)`, which rises steeply
# near 0 and flattens out by about ±50%. Refit on the same split (the same
# `random_state`), score it on the test set, and add its curves to the plot. Is
# it better?

# %% tags=["solution"]
X_tanh = np.column_stack([np.tanh(contrast / 25), probability_left])
Xt_train, Xt_test, _, _ = train_test_split(X_tanh, y, test_size=0.25, random_state=0)
model_tanh = LogisticRegression().fit(Xt_train, y_train)
print(f"test {model_tanh.score(Xt_test, y_test):.3f}")


def tanh_features(c, p_left):
    return np.column_stack([np.tanh(c / 25), np.full_like(c, p_left)])


fig, ax = plt.subplots()
ax.plot(grid, model_tanh.predict_proba(tanh_features(grid, 0.2))[:, 1], color="tab:red", label="tanh model, right blocks")
ax.plot(grid, model_tanh.predict_proba(tanh_features(grid, 0.8))[:, 1], color="tab:blue", label="tanh model, left blocks")
ax.plot(levels, frac_right_rb, "o", color="tab:red")
ax.plot(levels, frac_right_lb, "o", color="tab:blue")
ax.set_xlabel("contrast (%; negative = left)")
ax.set_ylabel("probability of choosing right")
ax.legend();

# %% [markdown] tags=["solution"]
# The curves now follow the data closely. The test score barely moves (0.851 to
# 0.858), because the score only counts which side is more likely, and both
# models agree on that for most trials. The probabilities are much better,
# which is what matters if you want to describe the mouse rather than just
# guess its choice.
#
# ### Does the block help? One split is not enough
#
# 1. Fit a model on the contrast alone (`X_train[:, :1]`: the slice keeps it
#    2-D) and score it on the test set. Does adding the block help?
# 2. One random split is one roll of the dice. `cross_val_score(model, X, y,
#    cv=5)` splits the data five ways, fits and scores on each, and returns the
#    five scores. Compare the two models on their mean cross-validated score.

# %% tags=["solution"]
model_contrast = LogisticRegression().fit(X_train[:, :1], y_train)
print(f"contrast only, test {model_contrast.score(X_test[:, :1], y_test):.3f}")

print(f"both, cross-validated      {cross_val_score(LogisticRegression(), X, y, cv=5).mean():.3f}")
print(f"contrast only, cross-validated {cross_val_score(LogisticRegression(), X[:, :1], y, cv=5).mean():.3f}")

# %% [markdown] tags=["solution"]
# On our one split the block seems to help (0.836 to 0.851). Cross-validated,
# the two are the same (about 0.855). The block only changes the choice on the
# hard trials near contrast 0, which are a small fraction of the session, so it
# hardly changes the score, even though its weight is large. Small differences
# between models on one split are often luck.
#
# ### Decode the choice from the neurons
#
# Can you predict the mouse's choice from what its neurons did? Use the total
# spike count of each of the 51 units after the stimulus as the features:
# `X_spikes = spike_counts[:, :, 10:].sum(axis=2).T`, shape (533, 51).
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
# The neurons predict the choice about 68% of the time, against about 50% for
# both controls, so they carry real information about it, and only once the
# stimulus has appeared. The five folds range from about 0.52 to 0.77, so
# report the spread as well as the mean. These are somatosensory and insular
# neurons, and the mouse starts turning the wheel within the window, so they
# may well be signalling the movement rather than the decision. Telling those
# apart needs a better design, not a better model.
