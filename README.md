# Introduction to Scientific Python

A two-day course (3 hours a day). Day 1 covers environments, NumPy and SciPy; Day 2 covers pandas, Matplotlib, seaborn and scikit-learn, ending with pointers to scikit-image, xarray, dask and napari. Each block mixes short runs of slides with live coding, then ends with exercises in a Jupyter notebook.

The exercises follow one real experiment from start to finish: a Neuropixels recording from a mouse doing a visual decision task, from the [International Brain Laboratory](https://www.internationalbrainlab.com/) via the [DANDI archive](https://dandiarchive.org/dandiset/000409). Students index its voltage traces by time and recording site, mask channels by brain area, compute the mouse's psychometric curve without a loop, filter mains hum out of the LFP, summarise its trials and neurons with pandas, plot them, and predict the mouse's choices with a logistic regression.

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

## Schedule

| Day 1 | min | Day 2 | min |
|---|---|---|---|
| Welcome, setup check and help | 10 | Recap | 5 |
| Packages and environments | 30 | pandas | 60 |
| NumPy | 80 | Matplotlib | 45 |
| *Break* | 10 | *Break* | 10 |
| SciPy | 30 | seaborn | 20 |
| Wrap-up | 5 | scikit-learn | 30 |
| | | Where to go next (slides only) and wrap-up | 10 |

| Notebook | Block | Main sources |
|---|---|---|
| `day1_01_getting_help` | Setup and help | SPL *Getting help* |
| `day1_02_environments` | Environments (terminal work) | conda and uv docs |
| `day1_03_numpy` | NumPy | An IBL Neuropixels recording (DANDI 000409), plus SPL Ex. 22 and 23 |
| `day1_04_scipy` | SciPy: filtering | After SPL Ex. 42, on the LFP (stretch: Ex. 40 and 41) |
| `day2_01_pandas` | pandas | SWC gapminder episodes 7 and 8, on the IBL trials and units tables |
| `day2_02_matplotlib` | Matplotlib | After SPL *Simple plot*, Ex. 28, 31 and 35, on the recording |
| `day2_03_seaborn` | seaborn | Live demo on the trials and units, after PDSH 4.14 |
| `day2_04_sklearn` | scikit-learn | Logistic regression of the mouse's choices; decoding them from spike counts (stretch) |

Each block has a core that most learners should finish in the time given, followed by stretch exercises for those who finish early. Each notebook recomputes what it needs from earlier blocks in its first cells, so a learner who did not finish one block can still start the next. scikit-image, xarray, dask and napari appear only on the "where to go next" slides, with code that is shown but not run, so they are not in the course environment.

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
- `tags=["raises-exception"]` marks a cell that is meant to fail. `tags=["no-execute"]` marks a cell the build skips (for example one that opens a window).

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
  - `ibl_session.h5`: a 1.9 MB slice of one session of the International Brain Laboratory's Brain Wide Map, [DANDI:000409](https://doi.org/10.48324/dandi.000409/0.260309.1324) (CC BY 4.0). `scripts/fetch_data.py` rebuilds it from DANDI.
  - `moonlanding.png` (Day 2 slides only): from SPL (CC BY 4.0)
  - gapminder CSVs (slides only): from Software Carpentry (CC BY 4.0)
