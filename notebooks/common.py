"""Shared setup for the analysis notebooks: paths, names and colours, the plot style, and loading results and datasets.

Every notebook imports what it needs from here, so that the same version or classifier has the same name and
colour everywhere, and every notebook computes the times in the same way. After changing this file, restart the
notebook kernel (or run the notebooks again) to pick up the change.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parent.parent          # quant-reproduction/
RESULTS_DIR = REPO / "results"                         # the results of our runs, used by notebooks 01-06
PUBLISHED_FILE = REPO / "original" / "results" / "accuracy_resamples_ucr142.csv"
DATA_TS = REPO / "data" / "ts"                         # the datasets (.ts files)
DATASET_LISTS = REPO / "improvedQuant" / "dataSetLists"

# ---------------------------------------------------------------------------------------------------- classifiers
CLASSIFIERS = ["extratrees", "xgboost", "catboost", "tabpfn"]
LABELS = {"extratrees": "ExtraTrees (original)", "xgboost": "XGBoost", "catboost": "CatBoost", "tabpfn": "TabPFN"}
COLORS = {"extratrees": "#2a78d6", "xgboost": "#eb6834", "catboost": "#1baf7a", "tabpfn": "#eda100"}

# ---------------------------------------------------------------------------------------------------- versions
# one colour per version, the same in every notebook; one hue per family (the notebooks explain the versions)
VERSION_COLORS = {
    "published": "#8c8b85",                                       # the authors' published results: grey
    "v0": "#8fc0f0", "v1": "#1f5fae",                             # our runs of the method: blues (v1 = the baseline)
    "v2-lagd": "#5a8f29", "v2-lagd-joint": "#a9cf7d",             # v2, QUANT + lag-d: greens
    "v2-multirocket": "#9085e9",                                  # v2, + lag-d + MultiRocket: purple
    "v2-rdst": "#0e8a8a",                                         # v2, + lag-d + RDST: teal
    "v2-mrhydra": "#e0609a",                                      # v2, + MR-Hydra: pink
}
BASELINE = "#8c8b85"     # grey for "v1" next to bars in the classifier colours (one bar per classifier)

# ---------------------------------------------------------------------------------------------------- plot style
TEXT, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
BETTER, EQUAL, WORSE = "#2a78d6", "#d9d8d2", "#e34948"           # better / equal / worse than the reference
DIVERGING = LinearSegmentedColormap.from_list("better_worse", [WORSE, "#f0efec", BETTER])


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
                "time_extra_train", "time_extra_test", "time_select", "time_fit", "time_predict"]


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
    d["transform_time"] = d[["time_transform_train", "time_transform_test", "time_lag_train", "time_lag_test",
                             "time_extra_train", "time_extra_test"]].sum(axis=1)
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
def short(name):
    """Two lines for long dataset names, split at a capital letter near the middle (for axis labels)."""
    if len(name) <= 14:
        return name
    cut = max(i for i, ch in enumerate(name[:len(name) // 2 + 3]) if ch.isupper() and i > 0)
    return name[:cut] + "-\n" + name[cut:]


def offsets(n, width):
    """x offsets of n bars of the given width, centred on their group."""
    return (np.arange(n) - (n - 1) / 2) * width


def signed(value):
    """+0.012 / -0.004 / 0.000 (never "-0.000")."""
    return "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"


def wdl_counts(differences):
    """How many differences are positive, zero and negative: (better, equal, worse)."""
    d = pd.Series(differences)
    return int((d > 1e-12).sum()), int((d.abs() <= 1e-12).sum()), int((d < -1e-12).sum())
