"""Shared setup for the analysis notebooks: paths, names and colours, the plot style, and loading results and datasets.

Every notebook imports what it needs from here, so that the same version or classifier has the same name and
colour everywhere, and every notebook computes the times in the same way. After changing this file, restart the
notebook kernel (or run the notebooks again) to pick up the change.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parent.parent          # quant-reproduction/
RESULTS_DIR = REPO / "results"                         # the results of our runs, used by notebooks 01-05
PUBLISHED_FILE = REPO / "original" / "results" / "accuracy_resamples_ucr142.csv"
DATA_TS = REPO / "data" / "ts"                         # the datasets (.ts files)
DATASET_LISTS = REPO / "improvedQuant" / "dataSetLists"

# ---------------------------------------------------------------------------------------------------- colours
# Every colour of the notebooks is defined here, once, and named by what it shows.

# the text, axes and grid; better / equal / worse than a reference (changes, better / equal / worse counts)
TEXT, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
BETTER, EQUAL, WORSE = "#2a78d6", "#d9d8d2", "#e34948"
WARNING = "#eda100"                                     # a value that needs attention (e.g. outside the expected noise)

# the classifiers on QUANT's features (notebooks 02, 03); ExtraTrees = QUANT's own classifier, the colour of v1
CLASSIFIERS = ["extratrees", "xgboost", "catboost", "tabpfn"]
LABELS = {"extratrees": "ExtraTrees (original)", "xgboost": "XGBoost", "catboost": "CatBoost", "tabpfn": "TabPFN"}
COLORS = {"extratrees": "#1f5fae", "xgboost": "#eb6834", "catboost": "#1baf7a", "tabpfn": "#eda100",
          "tabpfn default": "#f6d488"}                  # TabPFN with its default settings (8 passes, 500 features)

# one colour per result, the same in every notebook
VERSION_COLORS = {
    "published": "#8c8b85",                             # the authors' published results: grey
    "v0": "#8fc0f0",                                    # our run of the authors' code: light blue
    "v1": COLORS["extratrees"],                         # QUANT + ExtraTrees, the baseline: blue
    "v1 all cores": "#9cc3ef",                          # the same on all CPU cores (time comparison)
    "v1 - v0": "#6b4fbb",                               # notebook 01: the difference of our two runs
    "v3": "#4a3aa7",                                    # the cheap TabPFN: purple
    "v4": "#8e1b4f",                                    # the supervised extractor, the final version: wine
}
OTHERS = "#b4b2a9"                                      # results that are not ours (the published classifiers)

# groups of datasets: the two development lists (notebooks 02, 03), development vs held out (notebook 05)
GROUP_COLORS = {"small": "#9cc3ef", "medium": "#1f5fae", "development": "#c9c7bf", "held out": "#6b4fbb", "all": MUTED}

# the phases of a run's time (notebook 01)
PHASE_COLORS = {"transform train": "#4a3aa7", "transform test": "#9085e9", "fit": "#008300", "predict": "#8c8b85"}

# what the supervised extractor does with a candidate feature (notebook 04)
OUTCOME_COLORS = {"kept": BETTER, "kept (weak)": "#9cc3ef", "over budget": "#c9c7bf", "below the ANOVA cut": GRID,
                  "near-copy": WARNING, "irrelevant": WORSE}


def style():
    """The plot style of the notebooks: no top and right spines, a light horizontal grid, bold left titles."""
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })


# ---------------------------------------------------------------------------------------------------- results
TIME_COLUMNS = ["time_transform_train", "time_transform_test", "time_lag_train", "time_lag_test",
                "time_select", "time_fit", "time_predict"]


def load_runs(path):
    """A results file of the improvedQuant scripts, one row per run, with three time columns added.

    Time columns a version does not have (e.g. the lag-d transform in v1) are set to 0. Added: fit_predict (fit +
    predict), transform_time (all feature transforms) and total_time (transforms + feature selection + fit +
    predict), in seconds.
    """
    d = pd.read_csv(path)
    for column in TIME_COLUMNS:
        d[column] = d[column].fillna(0.0) if column in d else 0.0
    d["fit_predict"] = d["time_fit"] + d["time_predict"]
    d["transform_time"] = d[["time_transform_train", "time_transform_test", "time_lag_train", "time_lag_test"]].sum(axis=1)
    d["total_time"] = d["transform_time"] + d["time_select"] + d["fit_predict"]
    return d


# ---------------------------------------------------------------------------------------------------- datasets
def read_split(name, split):
    """(number of cases, series length, set of class labels) of <name>_<split>.ts, split = "TRAIN" or "TEST"."""
    lines = (DATA_TS / name / f"{name}_{split}.ts").read_text(encoding="utf-8").splitlines()
    start = [i for i, line in enumerate(lines) if line.strip().lower() == "@data"][0]
    data = [line for line in lines[start + 1:] if line.strip()]
    return len(data), len(data[0].split(":")[0].split(",")), {line.split(":")[-1].strip() for line in data}


def count_test_cases(name):
    """Number of series in the test set of a dataset."""
    return read_split(name, "TEST")[0]


def dataset_list(name):
    """The datasets of a list in improvedQuant/dataSetLists, e.g. dataset_list("smoke-medium")."""
    return (DATASET_LISTS / f"{name}.txt").read_text(encoding="utf-8").split()


# ---------------------------------------------------------------------------------------------------- small helpers
def signed(value):
    """+0.012 / -0.004 / 0.000 (never "-0.000")."""
    return "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"


def wdl_counts(differences):
    """How many differences are positive, zero and negative: (better, equal, worse)."""
    d = pd.Series(differences)
    return int((d > 1e-12).sum()), int((d.abs() <= 1e-12).sum()), int((d < -1e-12).sum())
