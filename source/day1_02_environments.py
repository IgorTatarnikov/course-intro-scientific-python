# %% [markdown]
# # Day 1 · Packages and environments
#
# **Block:** Packages and environments (25 min, including slides and a live demo)
#
# These exercises happen in a **terminal**, not in this notebook. Keep the
# notebook open as your instructions. On Windows use the "Miniforge Prompt";
# on macOS and Linux use any terminal.
#
# Neither Scientific Python Lectures nor the Python Data Science Handbook covers
# environments, so these exercises follow the tools' own guides:
# [conda: getting started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
# and [uv: working on projects](https://docs.astral.sh/uv/guides/projects/).
#
# > **Do not touch the course environment** (`scientific-python`). Everything
# > below builds throwaway environments, which you delete at the end.

# %% [markdown]
# ## Part 1: conda (12 min)
#
# 1. Create an environment called `scratch` with Python 3.12 and **pandas**, taking
#    packages only from the **conda-forge** channel.
# 2. Activate it and check which pandas version you got:
#    `python -c "import pandas; print(pandas.__version__)"`
# 3. You never asked for NumPy. Is it installed? Find out with `conda list`.
#    Why is it there?
# 4. Export the environment to a file called `scratch.yml`. Try exporting it
#    twice, with and without `--from-history`, and compare the two files.
#    Which one mentions NumPy? Which would you put in a repository, and why?
# 5. Deactivate, then build a **second** environment called `scratch-copy` from
#    `scratch.yml`.
# 6. List your environments to check that both exist, then remove both.
#
# Hints: `conda create --help`, `conda env export --help`, `conda env --help`.

# %% [markdown] tags=["answer"]
# ```bash
# # 1. Create
# conda create -n scratch -c conda-forge --override-channels python=3.12 pandas
#
# # 2. Activate and check
# conda activate scratch
# python -c "import pandas; print(pandas.__version__)"
#
# # 3. NumPy came along as a dependency of pandas
# conda list numpy
#
# # 4. Export, two ways
# conda env export > scratch-full.yml
# conda env export --from-history > scratch.yml
#
# # 5. Rebuild under a new name
# conda deactivate
# conda env create -n scratch-copy -f scratch.yml
#
# # 6. Check, then clean up
# conda env list
# conda env remove -n scratch
# conda env remove -n scratch-copy
# ```
#
# pandas is built on NumPy, so conda's solver installed NumPy (and pandas' other
# dependencies) for you.
#
# `scratch-full.yml` lists every package, NumPy included, with its exact build string, and
# those builds are often specific to your operating system. It is a precise
# record of *this* machine, but often fails to solve on a colleague's laptop.
# `--from-history` keeps only what you asked for (`python=3.12`, `pandas`), so it
# is portable. That is the one to commit, ideally with versions pinned the way
# the course's own `environment.yml` does.
#
# Both exports end with a `prefix:` line holding the path on *your* machine.
# Delete it before sharing the file. Also check that `channels:` lists
# `conda-forge`; add it by hand if your conda version left it out.

# %% [markdown]
# ## Part 2: uv (10 min)
#
# uv manages *projects*. A project is a folder with a `pyproject.toml`
# describing what you need, a `uv.lock` recording exactly what was installed, and
# a `.venv` folder holding the environment itself.
#
# 1. Make a new project: `uv init uv-demo`, then `cd uv-demo`. Look at the files
#    it created.
# 2. Add **SciPy** as a dependency. What changed in `pyproject.toml`? What new
#    files appeared? Is NumPy listed in `pyproject.toml`, in `uv.lock`, or both?
# 3. Replace the contents of `main.py` with:
#
#    ```python
#    import scipy
#    from scipy import constants
#
#    print("scipy", scipy.__version__, "speed of light:", constants.c)
#    ```
#
#    and run it **through uv**, without activating anything.
# 4. Delete the `.venv` folder completely. Recreate the environment from the
#    lockfile and run the script again. Is it the same SciPy version?
#
# Hints: `uv add`, `uv run`, `uv sync`.

# %% [markdown] tags=["answer"]
# ```bash
# uv init uv-demo
# cd uv-demo              # pyproject.toml, main.py, README.md, .python-version, .gitignore
#
# uv add scipy            # adds "scipy>=..." to dependencies; creates uv.lock and .venv
# uv run main.py          # creates/updates .venv if needed, then runs inside it
#
# rm -rf .venv            # on Windows: rmdir /s .venv
# uv sync                 # rebuilds .venv from uv.lock: same versions as before
# uv run main.py
# ```
#
# NumPy is only in `uv.lock`: `pyproject.toml` lists what you asked for, and the
# lockfile records every package that was installed, dependencies included.
#
# `uv run` would also have recreated `.venv` on its own: it always syncs the
# environment with the lockfile before running.

# %% [markdown]
# ## Part 3: compare
#
# We debrief these together at the end of the block. If you finish early, discuss them
# with your neighbour first.
#
# * Where does each tool keep the environment? (Hint: `conda env list` and `ls -a`.)
# * Which file would you commit so a colleague can reproduce your setup exactly?
# * conda can install non-Python things (compilers, CUDA, Qt, GDAL). uv installs
#   only Python packages from PyPI. When would that matter for your work?

# %% [markdown] tags=["answer"]
# * conda keeps named environments in one central folder (for example
#   `~/miniforge3/envs/scratch`), and you can use them from any directory. uv keeps
#   the environment in `.venv` **inside the project folder**, so the environment
#   belongs to that project.
# * For uv, commit `pyproject.toml` **and** `uv.lock`, but never `.venv`. For conda,
#   commit `environment.yml`. A plain `environment.yml` is not a lockfile;
#   [conda-lock](https://github.com/conda/conda-lock) or
#   [pixi](https://pixi.sh) (which writes `pixi.lock`) add one.
# * Whenever a package needs non-Python libraries: GPU stacks, Qt (napari),
#   geospatial libraries, MPI, or anything you would otherwise compile. pixi
#   combines conda-forge packages with a uv-style project workflow.

# %% [markdown]
# ## Stretch: pixi
#
# If you finish early and have [pixi](https://pixi.sh) installed: `pixi init
# pixi-demo`, `pixi add python numpy`, `pixi run python -c "import numpy"`. Look
# at `pixi.toml` and `pixi.lock`. Which of the two workflows above does it
# resemble?
