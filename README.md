# Introduction to Scientific Python

A two-day course (3 hours a day) covering environments, NumPy, pandas, Matplotlib, seaborn, SciPy and scikit-image, with a short tour of xarray, dask and napari. Each block mixes short runs of slides with live coding (Day 1 so far), then ends with exercises in a Jupyter notebook.

## For learners

Do this **before Day 1**:

1. Install [Miniforge](https://conda-forge.org/download/).
2. Download this repository (`git clone`, or *Code → Download ZIP* on GitHub).
3. In a terminal, from the repository folder:

   ```sh
   conda env create -f environment.yml
   conda activate scientific-python
   python check_setup.py
   ```

   Every line should say `OK`. If not, send the output to the instructors.

4. On the day, run `jupyter lab` from the repository folder and open the notebooks in `notebooks/`.

napari opens a desktop window, so please use your own laptop rather than JupyterHub or a remote server.

## Schedule

| Day 1 | min | Day 2 | min |
|---|---|---|---|
| Welcome, setup check and help | 10 | Matplotlib | 45 |
| Packages and environments | 30 | seaborn | 15 |
| NumPy | 75 | SciPy | 25 |
| *Break* | 10 | *Break* | 10 |
| pandas | 55 | scikit-image: counting nuclei | 45 |
| Wrap-up | 5 | Whirlwind tour: xarray, dask, napari | 35 |
| | | Wrap-up | 5 |

| Notebook | Block | Main sources |
|---|---|---|
| `day1_01_getting_help` | Setup and help | SPL *Getting help* |
| `day1_02_environments` | Environments (terminal work) | conda and uv docs |
| `day1_03_numpy` | NumPy | SPL Ex. 22–24 and 26, plus indexing, views-and-copies and in-place tasks |
| `day1_04_pandas` | pandas | SWC gapminder, episodes 7 and 8 |
| `day2_01_matplotlib` | Matplotlib | SPL *Simple plot*, Ex. 28 and 31, *Framing a Face* |
| `day2_02_seaborn` | seaborn | Live demo on gapminder, after PDSH 4.14 |
| `day2_03_scipy` | SciPy | SPL Ex. 39 and 42 (stretch: 40 and 41) |
| `day2_04_scikit_image` | scikit-image | Nuclei segmentation (stretch: SPL coins exercises) |
| `day2_05_xarray`, `day2_06_dask`, `day2_07_napari` | Whirlwind tour | Each project's tutorial |

Compared with the original three-day outline in `plan.md`, the two-day version moves these exercises to **stretch** (they are still in the notebooks): NumPy Ex. 24, pandas *Many Ways of Access*, the Matplotlib annotation step (now optional), SciPy Ex. 40 and 41, and the SPL coins exercises. The environments comparison is now a debrief led by the instructor, and the four "one slide each" packages share one slide.

## Repository layout

```
day1.qmd, day2.qmd, index.qmd   slide decks (Quarto + reveal.js)
source/                         single source for every notebook (Jupytext percent scripts)
notebooks/                      exercise notebooks (generated: solutions stripped)
solutions/                      solution notebooks (generated)
data/                           every dataset used, so nothing downloads in class
environment.yml, check_setup.py the learners' environment and its check
scripts/build_notebooks.py      builds notebooks/ and solutions/ from source/
scripts/fetch_data.py           rebuilds data/ from the original sources
```

## For instructors

### Editing notebooks

Edit the files in `source/`, never the generated `.ipynb` files. In a source file:

- `# %% tags=["solution"]` marks a code cell as a solution. The exercise version gets an empty `# Your code here` cell. A markdown cell tagged `solution` (an explanation) is dropped from the exercise version.
- `# %% [markdown] tags=["answer"]` marks the answer to a question. The exercise version shows *Your answer here.*
- `# BEGIN SOLUTION` / `# END SOLUTION` inside a cell blanks just that part, for example a function body.
- `tags=["raises-exception"]` marks a cell that is meant to fail. `tags=["no-execute"]` marks a cell the build skips (napari).

Then rebuild, inside the course environment plus `jupytext`:

```sh
pip install jupytext          # once; not part of the learners' environment
python scripts/build_notebooks.py           # build and execute everything
python scripts/build_notebooks.py day2_04   # just one notebook
```

The build executes every solution notebook **and** every exercise notebook, so it fails if a provided cell depends on a solution the learner hasn't written yet. It also fails if any solution line leaks into an exercise notebook, or if a solution marker is malformed.

### Releasing solutions

`solutions/` is generated alongside `notebooks/`. To release solutions block by block, keep `solutions/` out of the branch learners clone (add it to `.gitignore`, or publish it from a separate branch), and push each file after its block.

### Slides

Install [Quarto](https://quarto.org/docs/get-started/). Point `QUARTO_PYTHON` at an environment with the course packages, then run:

```sh
export QUARTO_PYTHON=$(which python)   # with the course environment active
quarto render                          # renders index, day1 and day2 into build/
quarto preview day1.qmd                # live preview while editing
```

Some slides run code at render time and read from `data/`. CI installs `requirements.txt` and renders all decks; pushing a release tag deploys them to GitHub Pages (see `.github/workflows/render_and_deploy.yml`).

## Licences and attribution

- Exercises adapted from [Scientific Python Lectures](https://lectures.scientific-python.org) (CC BY 4.0) and Software Carpentry's [Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder/) (CC BY 4.0).
- The [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) is linked for reading only. Its text is CC BY-NC-ND, so none of it is copied here.
- Data:
  - `populations.txt` and `moonlanding.png`: from SPL (CC BY 4.0)
  - gapminder CSVs: from Software Carpentry (CC BY 4.0)
  - `face.png`: from `scipy.datasets`
  - `human_mitosis.png`: from `skimage.data`, CC0, courtesy of David Root, from Moffat *et al.*, *Cell* 2006
  - `air_temperature_daily.nc`: NCEP reanalysis via the xarray tutorial data, reduced to daily means
