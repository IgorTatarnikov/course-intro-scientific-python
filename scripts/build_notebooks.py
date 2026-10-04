"""Build exercise and solution notebooks from the single-source files in ``source/``.

Each ``source/*.py`` file is a Jupytext "percent" script. Solutions are marked
in one of two ways:

* a whole cell tagged ``solution``::

      # %% tags=["solution"]
      a = np.arange(1, 16).reshape(3, 5).T

  In the exercise notebook a code cell is replaced by an empty prompt and a
  markdown cell (an explanation that comes with the solution) is dropped.

* a markdown cell tagged ``answer``: the answer to a question asked in the
  instructions. In the exercise notebook it becomes an "*Your answer here.*" prompt.

* a block inside a cell, between marker comments::

      def f(a, b, c):
          # BEGIN SOLUTION
          return a**b - c
          # END SOLUTION

  In the exercise notebook the block becomes ``...  # your code here``.

Cells tagged ``no-execute`` (for example one that opens a desktop window) are kept
in both notebooks but skipped when the build executes them.

Usage, from the repository root, inside the course environment plus jupytext::

    python scripts/build_notebooks.py            # build and execute everything
    python scripts/build_notebooks.py day1_03    # only sources whose name contains "day1_03"
    python scripts/build_notebooks.py --no-execute

Executing checks three things: every solution notebook runs, every exercise
notebook runs (so no provided cell depends on a solution the learner has not
written yet), and no solution line leaks into an exercise notebook.
"""

import argparse
import copy
import re
import sys
import tempfile
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source"
EXERCISES = ROOT / "notebooks"
SOLUTIONS = ROOT / "solutions"

BEGIN = re.compile(r"^(\s*)# BEGIN SOLUTION\s*$")
END = re.compile(r"^\s*# END SOLUTION\s*$")

CODE_PROMPT = "# Your code here\n"
MARKDOWN_PROMPT = "*Your answer here.*"


def strip_markers(source: str) -> str:
    """Solution version: drop the marker lines, keep the code between them."""
    return "\n".join(
        line
        for line in source.splitlines()
        if not BEGIN.match(line) and not END.match(line)
    )


def blank_solutions(source: str) -> str:
    """Exercise version: replace each marked block by an ellipsis prompt."""
    out, inside = [], False
    for line in source.splitlines():
        if match := BEGIN.match(line):
            inside = True
            out.append(f"{match.group(1)}...  # your code here")
        elif END.match(line):
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("BEGIN SOLUTION without matching END SOLUTION")
    return "\n".join(out)


def clean_metadata(cell) -> None:
    tags = [t for t in cell.metadata.get("tags", []) if t not in ("solution", "answer")]
    if tags:
        cell.metadata["tags"] = tags
    else:
        cell.metadata.pop("tags", None)


def split(nb):
    """Return (exercise, solution) notebooks and the solution-only lines."""
    exercise, solution = copy.deepcopy(nb), copy.deepcopy(nb)
    secret_lines, provided_lines = set(), set()

    for ex_cell, sol_cell in zip(exercise.cells, solution.cells, strict=True):
        for line in sol_cell.source.splitlines():
            if "SOLUTION" in line and not (BEGIN.match(line) or END.match(line)):
                raise ValueError(f"Malformed solution marker: {line!r}")
        tags = sol_cell.metadata.get("tags", [])
        if "solution" in tags or "answer" in tags:
            secret_lines.update(sol_cell.source.splitlines())
            if ex_cell.cell_type == "code":
                ex_cell.source = CODE_PROMPT
            elif "answer" in tags:
                ex_cell.source = MARKDOWN_PROMPT
            else:
                ex_cell.metadata["drop"] = True
        elif any(BEGIN.match(line) for line in sol_cell.source.splitlines()):
            ex_cell.source = blank_solutions(sol_cell.source)
            kept = set(ex_cell.source.splitlines())
            secret_lines.update(
                line
                for line in strip_markers(sol_cell.source).splitlines()
                if line not in kept
            )
        else:
            provided_lines.update(sol_cell.source.splitlines())
        sol_cell.source = strip_markers(sol_cell.source)
        clean_metadata(ex_cell)
        clean_metadata(sol_cell)

    exercise.cells = [c for c in exercise.cells if not c.metadata.pop("drop", False)]

    # A line the instructions show on purpose is not a secret, and neither are
    # short or generic lines (blank, "plt.show()", a lone bracket).
    provided = "\n".join(provided_lines)
    secret_lines = {
        line.strip()
        for line in secret_lines
        if len(line.strip()) > 20
        and not line.strip().startswith("#")
        and line.strip() not in provided
    }
    return exercise, solution, secret_lines


def execute(nb, workdir: Path, label: str) -> bool:
    """Run a copy of the notebook (minus no-execute cells) in ``workdir``."""
    runnable = copy.deepcopy(nb)
    runnable.cells = [
        c for c in runnable.cells if "no-execute" not in c.metadata.get("tags", [])
    ]
    client = NotebookClient(
        runnable,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    )
    try:
        client.execute()
        print(f"    ran      {label}")
        return True
    except Exception as err:  # CellExecutionError carries the traceback
        print(f"    FAILED   {label}\n{err}")
        return False


def check_leaks(exercise, secret_lines, label: str) -> bool:
    text = "\n".join(c.source for c in exercise.cells)
    leaks = sorted(line for line in secret_lines if line in text)
    for line in leaks:
        print(f"    LEAK     {label}: {line}")
    return not leaks


def build(path: Path, run: bool) -> bool:
    nb = jupytext.read(path)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.pop("jupytext", None)
    exercise, solution, secrets = split(nb)

    name = path.stem + ".ipynb"
    print(f"  {path.name}")
    ok = check_leaks(exercise, secrets, name)

    if run:
        # Run in a scratch copy of the notebooks folder layout, so files the
        # exercises write (e.g. processed.csv) do not end up in the repository.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "notebooks"
            workdir.mkdir()
            (Path(tmp) / "data").symlink_to(ROOT / "data")
            (Path(tmp) / "check_setup.py").symlink_to(ROOT / "check_setup.py")
            ok &= execute(solution, workdir, f"solutions/{name}")
            ok &= execute(exercise, workdir, f"notebooks/{name}")

    nbformat.write(exercise, EXERCISES / name)
    nbformat.write(solution, SOLUTIONS / name)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filter", nargs="?", default="", help="substring of source names")
    parser.add_argument("--no-execute", action="store_true", help="skip execution")
    args = parser.parse_args()

    EXERCISES.mkdir(exist_ok=True)
    SOLUTIONS.mkdir(exist_ok=True)
    sources = sorted(p for p in SOURCE.glob("*.py") if args.filter in p.stem)
    if not sources:
        sys.exit(f"No sources in {SOURCE} match {args.filter!r}")

    if not args.filter:
        # A full build also removes notebooks whose source has been renamed or deleted.
        stems = {p.stem for p in sources}
        for folder in (EXERCISES, SOLUTIONS):
            for orphan in folder.glob("*.ipynb"):
                if orphan.stem not in stems:
                    print(f"  removing orphan {orphan.relative_to(ROOT)}")
                    orphan.unlink()

    results = [build(path, run=not args.no_execute) for path in sources]
    if not all(results):
        sys.exit("\nBuild finished with problems (see above).")
    print(f"\nBuilt {len(results)} notebook(s).")


if __name__ == "__main__":
    main()
