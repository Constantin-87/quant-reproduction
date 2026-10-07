# The improvements after v1 (v2 to v9; experiments.md lists every version with its command), for the four
# classifiers of v1. One script for all of them; the version is chosen by the options:
#   v2: QUANT's features + lag-d differences and/or an extra feature set, filtered on the training data;
#   v3: TabPFN's cost settings; v4: the lean extractor quant_tabpfn.py (--quant tabpfn);
#   v5 to v9: the supervised extractor quant_tabpfn.SupervisedExtractor (--quant supervised).
#
# Features, in two stages (feature selection on the training data only):
#   1. QUANT + lag-d differences x(t) - x(t-d), d in --lags (summarised with QUANT's own interval quantiles,
#      "dilation before the quantiles"; A4(a) of the plan):
#        --lag_filter separate : keep ALL QUANT features and the best --lag_keep lag-d features (default:
#                                QUANT's count)
#        --lag_filter joint    : filter QUANT + lag-d together back to --lag_keep features
#   2. + the best --extra_keep features of an extra feature set (default: QUANT's count); the stage-1 features are
#      kept as they are (a joint filter with ~50,000 extra features would drop almost all of them):
#        --extra mrhydra       : MultiRocket + Hydra (aeon), --multirocket_kernels for MultiRocket's size
#        --extra multirocket   : MultiRocket without Hydra (Hydra made v2 slower and less accurate)
#        --extra rdst          : RDST, random dilated shapelets: 3 features per shapelet (smallest distance,
#                                where it occurs, how often), --rdst_shapelets shapelets
#        --extra none          : no extra feature set
#   --prune filter     : ANOVA F-score (both stages)
#   --prune importance : the extra features ranked by an ExtraTrees fit's feature importances (stage 2; stage 1
#                        uses the ANOVA filter)
#   --prune none       : no filtering (all features)
#   --extra_test selected : "train with many, predict with few": the training set gets all extra features (the
#                           filter must score all of them), the test set only the kept ones (MultiRocket:
#                           multirocket_selected.py; RDST: aeon's transform with the kept shapelets only)
#   --extra_test all      : compute all extra features for the test set too, then keep the columns (MR-Hydra
#                           always does this)
# The extra transforms come from aeon and are fitted on the training data only.
#
# The v2 versions (notebook 03) and their options:
#   v2-lagd          --lags 2,4,8,16 --extra none --lag_filter separate
#   v2-lagd-joint    --lags 2,4,8,16 --extra none --lag_filter joint
#   v2-multirocket   --lags 2,4,8,16 --extra multirocket --lag_filter separate --extra_test selected
#   v2-rdst          --lags 2,4,8,16 --extra rdst --lag_filter separate --extra_test selected
#   v2-mrhydra       --extra mrhydra --prune filter
#   (QUANT only      --lags none --extra none, the default: v3's runs on QUANT's own features)
#
# v3, TabPFN's cost settings (notebook 04): --tabpfn_estimators (passes), --tabpfn_features (feature budget),
# --tabpfn_context (at most this many training cases) and --tabpfn_cache (build TabPFN's model once).
#
# Built on improvedQuant/v1_classifiers/run.py, a modified copy of the authors' original/code/run.py (GPL-3.0).
# quant.py is unchanged; quant_tabpfn.py (--quant tabpfn) is a changed copy for TabPFN.
#
# Changes of A1 compared with the original run.py (all kept here):
#   - --classifier: extratrees (original), xgboost, catboost or tabpfn
#   - fixed random seed (--seed); the original run.py is unseeded
#   - metrics from predict_proba: accuracy, balanced accuracy, AUROC, log loss
#   - per-phase timers: transform train, fit, transform test, predict
#   - one CSV row written after every run (a crash does not lose earlier results;
#     a restart skips runs that are already in the file)
#   - a failed run (e.g. GPU out of memory) is recorded with its error instead of
#     stopping the whole experiment
#
# Example (from the quant-reproduction folder):
#   .venv-gpu\Scripts\python.exe improvedQuant\features\run.py -i data/ts -p data/indices
#       -d improvedQuant/dataSetLists/smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter separate
#       -o results/v2-lagd_smoke-small_allcores-gpu.csv
#   (needs .venv-gpu: aeon is installed there, and tabpfn needs CUDA PyTorch)

import argparse
import csv
import gc
import os
import re
import time
import warnings

# numpy, torch, sklearn etc. are imported after the arguments are read (see below),
# because the single-core setting (-j 1) must be in place before they load

# ==============================================================================
# == args ======================================================================
# ==============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_path")
parser.add_argument("-o", "--output_file", default = "results.csv")

parser.add_argument("-d", "--datasets")

parser.add_argument("-r", "--num_resamples", type = int, default = 30)
parser.add_argument("-p", "--resample_indices_path")

# QUANT's depth (levels of pieces) and div (one quantile per div points); default: 6 and 4 (original), 5 and 8 (tabpfn)
parser.add_argument("-m", "--depth", type = int, default = None)
parser.add_argument("-v", "--div", type = int, default = None)
# the feature extractor: "original" = the authors' quant.py, "tabpfn" = quant_tabpfn.py (fewer, less redundant
# features for TabPFN's small feature budget; its header lists the changes)
parser.add_argument("--quant", default = "original", choices = ["original", "tabpfn", "supervised"])
# --quant supervised (quant_tabpfn.SupervisedExtractor): the extractor itself returns at most --supervised_budget
# features, chosen on the training labels (relevance check + greedy selection without near-copies); views: QUANT's
# four, the lag-d differences of --lags, + --znorm (a shape-only view); extra candidates: --catch22, --templates
parser.add_argument("--supervised_budget", type = int, default = 250)
# limits relative to the dataset: fill the budget (past p >= 0.05, then finer pieces up to depth 6)
parser.add_argument("--fill_budget", action = "store_true")
# catch22 on a shortened copy (the mean of every k points), k chosen per dataset from the training series
parser.add_argument("--c22_shorten", action = "store_true")
# the budget relative to the dataset size: TabPFN's time grows with (training + test series) x features, so above
# 2000 series the budget shrinks in proportion (--supervised_budget x 2000 / series), never below --relative_min
parser.add_argument("--relative_budget", action = "store_true")
parser.add_argument("--relative_min", type = int, default = 100)
# the order of the greedy selection: anova = the ANOVA ranking; both = as --tabpfn_rank both (ANOVA keeps the budget
# + half of the rest, then the ExtraTrees importance ranks them)
parser.add_argument("--supervised_rank", default = "anova", choices = ["anova", "both"])
# the deepest pieces of the fill rounds: 6 = down to 1/32 of the series (v5 to v8), 5 = down to 1/16 (v9)
parser.add_argument("--supervised_max_depth", type = int, default = 6)
# the quantiles of QUANT's four views: 0 = min and max (--quantiles 2, v5 to v8); 0.2 = a fifth of QUANT's (one per 4
# points), odd so that the median stays in (v9; the lag-d views keep min and max)
parser.add_argument("--quantile_fraction", type = float, default = 0)
parser.add_argument("--znorm", action = "store_true")
parser.add_argument("--catch22", action = "store_true")
parser.add_argument("--templates", action = "store_true")
# --quant tabpfn only: a fixed number of quantiles per piece (e.g. 2 = min and max); 0 = one per div points
parser.add_argument("--quantiles", type = int, default = 0)
# how TabPFN's features are picked: "extratrees" (the most important in an ExtraTrees fit), "anova" (the highest
# ANOVA F-scores) or "both" (ANOVA drops the worse half beyond k, then ExtraTrees picks the k most important)
parser.add_argument("--tabpfn_rank", default = "extratrees", choices = ["extratrees", "anova", "both"])
# also write <output>_features.csv: how many features of each class (source, kind) are produced, kept by the lag-d
# filter and chosen by TabPFN, per dataset and resample
parser.add_argument("--feature_log", action = "store_true")

parser.add_argument("-e", "--num_estimators", type = int, default = 200)   # ExtraTrees and CatBoost
parser.add_argument("-f", "--max_features", default = "0.1")               # ExtraTrees only

# new in A1
CLASSIFIERS = ["extratrees", "xgboost", "catboost", "tabpfn"]
# one or more classifiers, e.g. "-c extratrees tabpfn", or "-c all"; they share the same QUANT features
parser.add_argument("-c", "--classifier", nargs = "+", default = ["extratrees"], choices = CLASSIFIERS + ["all"])
parser.add_argument("-s", "--seed", type = int, default = 0)
parser.add_argument("-j", "--n_jobs", type = int, default = -1)             # 1 = single CPU core
parser.add_argument("--device", default = "cuda", choices = ["cuda", "cpu"])  # for xgboost, catboost, tabpfn
parser.add_argument("--tabpfn_features", type = int, default = 500)        # TabPFN: keep only the k most important features
# TabPFN's ensemble size (number of forward passes that are averaged); 0 = TabPFN's default (8). v5: 2 passes
# with 200 features keep almost all of TabPFN's accuracy at about a fifth of the time
parser.add_argument("--tabpfn_estimators", type = int, default = 0)
# TabPFN's training context: at most this many training cases, a stratified random subset (0 = all of them);
# runs where the training set is not larger are recorded as "skipped" (they would repeat the run without it)
parser.add_argument("--tabpfn_context", type = int, default = 0)
# TabPFN's own model cache: build the model once per process instead of for every run (about 2 s per run, the
# same predictions); off by default, so the timings stay comparable with the files made without it
parser.add_argument("--tabpfn_cache", action = "store_true")
# GPU memory (6 GB): CatBoost needs memory in proportion to the number of features (42,081 features asked for
# 9.2 GB), and a CUDA out-of-memory error inside CatBoost kills the whole process. XGBoost does not fail but spills
# into (very slow) system memory and hangs: at 84,162 features the laptop crashed, 48,030 and 41,978 features hung,
# although 42,081 once ran. So both skip feature sets above this safe limit
parser.add_argument("--catboost_max_features", type = int, default = 14000)
parser.add_argument("--xgboost_max_features", type = int, default = 14000)

# stage 1: lag-d differences, as extra QUANT representations, e.g. "2,4,8,16" ("none" = no lag-d differences)
parser.add_argument("--lags", default = "none")
# stage 2: the extra feature set
parser.add_argument("--extra", default = "none", choices = ["none", "mrhydra", "multirocket", "rdst"])
# size of MultiRocket (mrhydra, multirocket): 6,250 kernels = aeon's default (49,728 features); 1,250 = the smaller
# variant of the MultiRocket paper
parser.add_argument("--multirocket_kernels", type = int, default = 6250)
# size of RDST: 10,000 shapelets = aeon's default (30,000 features)
parser.add_argument("--rdst_shapelets", type = int, default = 10000)
# feature selection on the training data: "filter" (ANOVA F-score), "importance" (stage 2 ranked by ExtraTrees
# feature importances; stage 1 still uses the ANOVA filter) or "none" (all features)
parser.add_argument("--prune", default = "filter", choices = ["none", "filter", "importance"])
# stage 1: how many lag-d (separate) or QUANT + lag-d (joint) features to keep; 0 = as many as QUANT has
parser.add_argument("--lag_keep", type = int, default = 0)
# stage 2: how many extra features to add; 0 = as many as QUANT has
parser.add_argument("--extra_keep", type = int, default = 0)
# stage 1: "separate" = keep all QUANT features, filter only the lag-d ones; "joint" = filter QUANT + lag-d together
parser.add_argument("--lag_filter", default = "separate", choices = ["joint", "separate"])
# stage 2 on the test set: "selected" = compute only the kept extra features; "all" = all, then the kept columns
parser.add_argument("--extra_test", default = "selected", choices = ["all", "selected"])

args = parser.parse_args()

classifiers = CLASSIFIERS if "all" in args.classifier else list(dict.fromkeys(args.classifier))

if args.depth is None:
    args.depth = {"original": 6, "tabpfn": 5, "supervised": 4}[args.quant]
if args.div is None:
    args.div = 4 if args.quant == "original" else 8
if args.quant == "supervised" and args.quantiles == 0:
    args.quantiles = 2                                   # as v6: min and max

_max_features = float(args.max_features) if args.max_features not in ["sqrt"] else args.max_features

# single core (-j 1), as in the paper's timings: the numerical libraries read these
# variables when they are imported, so they are set before the imports below
if args.n_jobs == 1:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

if args.tabpfn_cache:
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"

import numpy as np
import torch

if args.quant == "original":
    from quant import IntervalModel, Quant
else:
    from quant_tabpfn import IntervalModel, Quant, SupervisedExtractor
# the fixed number of quantiles per piece exists only in quant_tabpfn.py (not in the authors' quant.py)
QUANTILES = {"num_quantiles": args.quantiles} if args.quant == "tabpfn" else {}
# the settings of the supervised extractor, as one short text for the results file, e.g. "z1-c1-t1-b250"
SUPERVISED = (f"z{int(args.znorm)}-c{int(args.catch22)}-t{int(args.templates)}-b{args.supervised_budget}-f{int(args.fill_budget)}-s{int(args.c22_shorten)}-r{int(args.relative_budget)}-n{args.relative_min}-k{args.supervised_rank}-d{args.supervised_max_depth}"
              + (f"-q{args.quantile_fraction:g}" if args.quantile_fraction > 0 else "")
              if args.quant == "supervised" else "")
assert args.quantile_fraction == 0 or args.quant == "supervised", "--quantile_fraction needs --quant supervised"

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

if args.n_jobs == 1:
    torch.set_num_threads(1)            # the QUANT transform (and TabPFN on the CPU) on one core
    torch.set_num_interop_threads(1)

print(f"-- args ".ljust(80, "-"))
[print(f"{k}: {v}") for k, v in vars(args).items()]
print("-" * 80)

# ==============================================================================
# == file handling =============================================================
# ==============================================================================

# from <https://github.com/aeon-toolkit/aeon/blob/main/aeon/datasets/_data_loaders.py>

def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["timestamps", "missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        line = re.sub(r"\s+", " ", line)
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise OSError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise OSError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname":
                    meta_data[key] = tokens[1]
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise OSError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise OSError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data

def _load_data(file, meta_data, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    n_timepoints = 0
    y_values = []
    target = False
    if meta_data["classlabel"] or meta_data["targetlabel"]:
        target = True
    for line in file:
        line = line.strip().lower()
        line = line.replace("nan", replace_missing_vals_with)
        line = line.replace("?", replace_missing_vals_with)
        if "timestamps" in meta_data and meta_data["timestamps"]:
            channels = _get_channel_strings(line, target, replace_missing_vals_with)
        else:
            channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if target:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                n_timepoints = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise OSError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise OSError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = n_timepoints
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                equal_length = meta_data["equallength"]
                raise OSError(
                    f"channel {i} in case {n_cases} has a different number of "
                    f"observations to the other channels. "
                    f"Saw {current_length} in the first channel but"
                    f" {len(data_series)} in the channel {i}. The meta data "
                    f"specifies equal length == {equal_length}. But even if series "
                    f"length are unequal, all channels for a single case must be the "
                    f"same length"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if target:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data

def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, n_timepoints) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"]:
            data = data.squeeze()
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)
    if return_meta_data:
        return data, y, meta_data
    return data, y

# ==============================================================================
# == run =======================================================================
# ==============================================================================

# ==============================================================================
# == classifiers ===============================================================
# ==============================================================================

class SkipRun(Exception):
    """A classifier cannot run on this dataset because of a known limit (recorded as "skipped")."""

def make_extratrees(seed):
    # the original QUANT classifier (paper, Section 3.4), now with a seed
    return ExtraTreesClassifier(
        n_estimators = args.num_estimators,
        max_features = _max_features,
        criterion = "entropy",
        n_jobs = args.n_jobs,
        random_state = seed,
    )

def make_classifier(name, n_classes, seed):
    """Return a new classifier. Settings are fixed in advance (library defaults), never tuned on test data."""

    if name == "extratrees":
        return make_extratrees(seed)

    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            tree_method = "hist",
            device = args.device,
            n_jobs = args.n_jobs,
            random_state = seed,
        )

    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations = args.num_estimators,   # 200 trees, as QUANT's ExtraTrees (CatBoost's default of 1000 is ~5x slower)
            loss_function = "MultiClass" if n_classes > 2 else "Logloss",
            task_type = "GPU" if args.device == "cuda" else "CPU",
            thread_count = args.n_jobs,
            random_seed = seed,
            verbose = 0,
            allow_writing_files = False,   # do not create a catboost_info folder
        )

    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        if args.tabpfn_estimators > 0:
            return TabPFNClassifier(n_estimators = args.tabpfn_estimators, device = args.device, random_state = seed)
        return TabPFNClassifier(device = args.device, random_state = seed)

def select_features(Z_TR, Y_TR, k, seed):
    """Indices of the k best features for TabPFN, on the TRAINING data only (--tabpfn_rank):
    extratrees = the most important in an ExtraTrees fit, anova = the highest ANOVA F-scores,
    both = ANOVA drops the worse half of the features beyond k, then the k most important of the rest in an
    ExtraTrees fit (e.g. 450 features, k = 300: ANOVA keeps 375, ExtraTrees picks 300)."""
    n = Z_TR.shape[1]
    if n <= k:
        return np.arange(n)
    if args.tabpfn_rank == "anova":
        return best_k(Z_TR, Y_TR, k)
    candidates = best_k(Z_TR, Y_TR, k + (n - k) // 2) if args.tabpfn_rank == "both" else np.arange(n)
    ranker = make_extratrees(seed).fit(Z_TR[:, candidates], Y_TR)
    return np.sort(candidates[np.argsort(ranker.feature_importances_)[::-1][:k]])

def context_subset(Y_TR, cap, seed):
    """A stratified random subset of cap training cases for TabPFN (every class keeps its share)."""
    subset, _ = train_test_split(np.arange(len(Y_TR)), train_size = cap, stratify = Y_TR, random_state = seed)
    return np.sort(subset)

def compute_metrics(Y_TE, proba, n_classes):
    """Accuracy, balanced accuracy, AUROC and log loss from the class probabilities."""
    predictions = proba.argmax(axis = 1)    # same as predict() for all four classifiers
    labels = np.arange(n_classes)
    try:
        if n_classes == 2:
            auroc = roc_auc_score(Y_TE, proba[:, 1])
        else:
            auroc = roc_auc_score(Y_TE, proba, multi_class = "ovr", average = "weighted", labels = labels)
    except ValueError:              # e.g. a class that does not occur in this test set
        auroc = np.nan
    return {
        "accuracy": accuracy_score(Y_TE, predictions),
        "balanced_accuracy": balanced_accuracy_score(Y_TE, predictions),
        "auroc": auroc,
        "log_loss": log_loss(Y_TE, np.clip(proba, 1e-15, 1), labels = labels),
    }

# ==============================================================================
# == run =======================================================================
# ==============================================================================

COLUMNS = [
    "dataset", "resample", "classifier", "lags", "extra", "multirocket_kernels", "rdst_shapelets", "prune",
    "lag_filter", "extra_test", "quant", "quantiles", "supervised", "tabpfn_rank", "tabpfn_estimators", "tabpfn_context", "tabpfn_cache", "seed", "cpu_cores", "device",
    "n_features", "n_features_quant", "n_features_lag", "n_features_extra",
    "n_kept_quant", "n_kept_lag", "n_kept_extra", "n_features_used", "n_train", "n_train_used",
    "accuracy", "balanced_accuracy", "auroc", "log_loss",
    "time_transform_train", "time_lag_train", "time_extra_train", "time_select", "time_fit",
    "time_transform_test", "time_lag_test", "time_extra_test", "time_predict", "status",
]

# --lags auto (--quant supervised only): a ladder of lags, the extractor picks per dataset (SupervisedExtractor.lag_list)
LAGS = [] if args.lags.lower() == "none" else "auto" if args.lags == "auto" else [int(d) for d in args.lags.split(",")]
assert LAGS != "auto" or args.quant == "supervised", "--lags auto needs --quant supervised"

REPRESENTATIONS = ["series", "change (smoothed)", "change of change", "rhythm (|FFT|)"]   # QUANT's four views

def feature_kinds(model):
    """The kind of every column an IntervalModel writes, in its column order (both extractors)."""
    kinds = []
    for a, b in model.intervals:
        n = int(b - a)
        fixed = getattr(model, "num_quantiles", 0)       # quant_tabpfn.py with --quantiles
        n_quantiles = 1 if n == 1 else (min(fixed, n) if fixed > 0 else 1 + (n - 1) // model.div)
        if n == 1 or n_quantiles == 1:
            kinds.append("quantile")
        elif args.quant == "original":                 # every second quantile minus the piece's average
            kinds += ["quantile" if q % 2 == 0 else "quantile - mean" for q in range(n_quantiles)]
        else:
            kinds += ["quantile"] * n_quantiles
        if args.quant == "tabpfn":                     # quant_tabpfn.py: + mean, spread and trend of the piece
            kinds += ["mean", "spread", "trend"]
    return kinds

def supervised_budget(n_series):
    """--relative_budget: the extractor's budget for a dataset with n_series training + test series."""
    if not args.relative_budget or n_series <= 2000:
        return args.supervised_budget
    return max(args.relative_min, round(args.supervised_budget * 2000 / n_series))

def log_selection(dataset, i, log):
    """--quant supervised: what happened to every candidate, counted per (source, kind, level, position, outcome,
    copy_of), appended to <output>_selection.csv (notebook 05)."""
    path = args.output_file[:-4] + "_selection.csv"
    new_file = not os.path.exists(path)
    counts = {}
    for m in log:
        k = (m["source"], m["kind"], m["level"], m["position"], m["outcome"], m["copy_of"])
        counts[k] = counts.get(k, 0) + 1
    with open(path, "a", newline = "") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["dataset", "resample", "source", "kind", "level", "position", "outcome", "copy_of", "count"])
        for k, count in sorted(counts.items()):
            writer.writerow([dataset, i, *k, count])

def log_features(dataset, i, step, names):
    """Append the number of features of each (source, kind) to <output>_features.csv."""
    path = args.output_file[:-4] + "_features.csv"
    new_file = not os.path.exists(path)
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    with open(path, "a", newline = "") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["dataset", "resample", "step", "source", "kind", "count"])
        for (source, kind), count in sorted(counts.items()):
            writer.writerow([dataset, i, step, source, kind, count])

def lag_features(X_TR, X_TE):
    """Version 3: QUANT's interval quantiles on lag-d differences x(t) - x(t-d), d in --lags.

    Uses QUANT's own IntervalModel (same dyadic intervals, quantiles and mean subtraction as QUANT), so each
    lag-d difference is simply one more QUANT representation. Lags too long for the series are skipped.
    Returns the train and test features and the transform times.
    """
    F_TR, F_TE, t_train, t_test, names = [], [], 0.0, 0.0, []
    for d in LAGS:
        if X_TR.shape[-1] <= d + 1:
            continue
        t0 = time.perf_counter()
        D_TR = X_TR[..., d:] - X_TR[..., :-d]
        model = IntervalModel(input_length = D_TR.shape[-1], depth = args.depth, div = args.div, **QUANTILES)
        F_TR.append(model.fit_transform(D_TR, None).numpy())
        names += [(f"lag-{d}", kind) for kind in feature_kinds(model)]
        t1 = time.perf_counter()
        F_TE.append(model.transform(X_TE[..., d:] - X_TE[..., :-d]).numpy())
        t_train, t_test = t_train + (t1 - t0), t_test + (time.perf_counter() - t1)
    if not F_TR:
        return None, None, 0.0, 0.0, []
    return np.hstack(F_TR), np.hstack(F_TE), t_train, t_test, names

def extra_fit(X_TR, Y_TR, seed):
    """Stage 2: the extra feature set (--extra), fitted on the training data only; all its training features.

    Returns (fitted transform, training features as float32, time).
      mrhydra     : Hydra + MultiRocket, columns [Hydra | MultiRocket] (as MR-Hydra)
      multirocket : MultiRocket without Hydra
      rdst        : RDST, which uses the class labels to set its thresholds
    """
    from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform

    X_TR = X_TR.numpy().astype(np.float64)       # aeon expects numpy arrays (n_cases, n_channels, length)
    t0 = time.perf_counter()
    if args.extra == "mrhydra":
        hydra = HydraTransformer(n_jobs = args.n_jobs, random_state = seed, output_type = "numpy")
        multirocket = MultiRocket(n_kernels = args.multirocket_kernels, n_jobs = args.n_jobs, random_state = seed)
        H, M = hydra.fit_transform(X_TR), multirocket.fit_transform(X_TR)
        E_TR = np.concatenate([H, M], axis = 1, dtype = np.float32)
        model = (hydra, multirocket)
    elif args.extra == "multirocket":
        model = MultiRocket(n_kernels = args.multirocket_kernels, n_jobs = args.n_jobs, random_state = seed)
        E_TR = model.fit_transform(X_TR).astype(np.float32)
    else:
        model = RandomDilatedShapeletTransform(max_shapelets = args.rdst_shapelets, n_jobs = args.n_jobs,
                                               random_state = seed)
        E_TR = model.fit_transform(X_TR, Y_TR).astype(np.float32)
    return model, E_TR, time.perf_counter() - t0

def extra_test(model, X_TE, keep):
    """The extra features of the test set, only the columns in keep (None = all).

    --extra_test selected: compute only the kept columns (MultiRocket: multirocket_selected.py; RDST: shapelet i gives
                           columns 3i, 3i+1, 3i+2, so aeon's own transform function is called with the kept shapelets
                           only). The same values as computing all of them and keeping the columns.
    --extra_test all (and always for mrhydra): compute all features, then keep the columns (the column copy counts as
                           selection time, as in the recorded runs, so it is timed separately)
    Returns (E_TE, transform time, selection time).
    """
    X_TE = X_TE.numpy().astype(np.float64)
    t0 = time.perf_counter()
    if args.extra_test == "selected" and keep is not None and args.extra == "multirocket":
        from multirocket_selected import transform_selected
        E_TE = transform_selected(model, X_TE, keep, args.n_jobs)
        return E_TE, time.perf_counter() - t0, 0.0
    if args.extra_test == "selected" and keep is not None and args.extra == "rdst":
        from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import dilated_shapelet_transform
        shapelets = np.unique(keep // 3)                                 # the shapelets the kept columns come from
        subset = tuple(values[shapelets] for values in model.shapelets_)  # only their parameters
        E_TE = dilated_shapelet_transform(X_TE, subset)                  # 3 columns per kept shapelet
        E_TE = E_TE[:, 3 * np.searchsorted(shapelets, keep // 3) + keep % 3]
        E_TE = np.nan_to_num(E_TE, nan = 0.0, posinf = 0.0, neginf = 0.0).astype(np.float32)   # as aeon does
        return E_TE, time.perf_counter() - t0, 0.0
    if args.extra == "mrhydra":
        hydra, multirocket = model
        E_TE = np.concatenate([hydra.transform(X_TE), multirocket.transform(X_TE)], axis = 1, dtype = np.float32)
    else:
        E_TE = model.transform(X_TE).astype(np.float32)
    t1 = time.perf_counter()
    if keep is not None:
        E_TE = E_TE[:, keep]
    return E_TE, t1 - t0, time.perf_counter() - t1

def best_k_importance(F_TR, Y_TR, k, seed):
    """Indices of the k features with the highest ExtraTrees feature importance on the training data (sorted)."""
    if k >= F_TR.shape[1]:
        return np.arange(F_TR.shape[1])
    scores = make_extratrees(seed).fit(F_TR, Y_TR).feature_importances_
    return np.sort(np.argsort(scores)[::-1][:k])

def best_k(F_TR, Y_TR, k):
    """Indices of the k features with the highest ANOVA F-score on the training data (sorted)."""
    from sklearn.feature_selection import f_classif

    if k >= F_TR.shape[1]:
        return np.arange(F_TR.shape[1])
    with warnings.catch_warnings(), np.errstate(all = "ignore"):
        warnings.simplefilter("ignore")          # constant features give divide-by-zero warnings
        scores, _ = f_classif(F_TR, Y_TR)
    scores = np.nan_to_num(scores, nan = 0.0)
    return np.sort(np.argsort(scores)[::-1][:k])

datasets = np.atleast_1d(np.loadtxt(f"{args.datasets}", "str"))   # atleast_1d: a list with one name works too

# results file: create it with a header, or resume from the runs already in it
os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok = True)
done = set()
fieldnames = COLUMNS
if os.path.exists(args.output_file):
    with open(args.output_file, newline = "") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or COLUMNS      # a file made by an older script keeps its own columns
        done = {(row["dataset"], int(row["resample"]), row["classifier"]) for row in reader}
    print(f"resuming: {len(done)} runs already in {args.output_file}")
else:
    with open(args.output_file, "w", newline = "") as f:
        csv.writer(f).writerow(COLUMNS)

# column names of the older scripts (v3_lagd_multirocket, v2_mrhydra) -> the name used here
OLD_NAMES = {"mr_test": "extra_test", "n_features_extra_kept": "n_kept_extra"}

def save(row):
    """Append one run to the results file, in the file's own column order (also for files of the older scripts)."""
    for old, new in OLD_NAMES.items():
        if old in fieldnames and new in row:
            row[old] = row[new]
    with open(args.output_file, "a", newline = "") as f:
        csv.DictWriter(f, fieldnames, extrasaction = "ignore").writerow(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

# GPU warm-up: the first GPU fit also starts CUDA, which would inflate the first timing
for name in classifiers:
    if name != "extratrees" and args.device == "cuda":
        try:
            rng = np.random.default_rng(0)
            make_classifier(name, 2, args.seed).fit(rng.normal(size = (64, 8)), np.arange(64) % 2)
        except Exception as error:
            print(f"GPU warm-up failed for {name}: {type(error).__name__}: {error}")

# warm-up of the extra transform: aeon's transforms are compiled (numba) on their first call, which would
# otherwise be counted in the first dataset's time
if args.extra != "none":
    rng = np.random.default_rng(0)
    _model, _, _ = extra_fit(torch.tensor(rng.normal(size = (10, 1, 64)), dtype = torch.float32), np.arange(10) % 2,
                             args.seed)
    extra_test(_model, torch.tensor(rng.normal(size = (4, 1, 64)), dtype = torch.float32), np.arange(9))
    del _model
# the same for catch22 in the supervised extractor (--catch22)
if args.quant == "supervised" and args.catch22:
    rng = np.random.default_rng(0)
    _warm = SupervisedExtractor(catch22 = True, budget = 5)
    _warm.fit_transform(torch.tensor(rng.normal(size = (10, 1, 64)), dtype = torch.float32), np.arange(10) % 2)
    del _warm

Z_TR = Z_TE = None

for dataset in tqdm(datasets, ncols = 80):

    for i in range(args.num_resamples):

        todo = [name for name in classifiers if (dataset, i, name) not in done]
        if not todo:
            continue

        # free the previous run's feature matrices before building new ones
        # (on large datasets each one is several GB)
        Z_TR = Z_TE = None
        gc.collect()

        # == data (same as the original run.py) ================================

        _X_TR, _Y_TR = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TRAIN.ts")
        _X_TE, _Y_TE = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TEST.ts")

        if i == 0:
            X_TR = _X_TR
            X_TE = _X_TE
            Y_TR = _Y_TR
            Y_TE = _Y_TE
        else:
            _X = np.concatenate((_X_TR, _X_TE))
            _Y = np.concatenate((_Y_TR, _Y_TE))

            indices = np.arange(_X.shape[0])

            indices_TR = np.loadtxt(f"{args.resample_indices_path}/{dataset}/resample{i}Indices_TRAIN.txt", dtype = np.int32)
            indices_TE = np.setdiff1d(indices, indices_TR)

            X_TR = _X[indices_TR]
            X_TE = _X[indices_TE]

            Y_TR = _Y[indices_TR]
            Y_TE = _Y[indices_TE]

        X_TR = torch.tensor(X_TR.squeeze().astype(np.float32)).unsqueeze(1)
        X_TE = torch.tensor(X_TE.squeeze().astype(np.float32)).unsqueeze(1)

        le = LabelEncoder()

        Y_TR = le.fit_transform(Y_TR)
        Y_TE = le.transform(Y_TE)

        n_classes = len(le.classes_)

        # == transform (QUANT, see --quant), once, shared by all classifiers ===========

        try:
            if args.quant == "supervised":
                # change 7: the extractor returns the chosen features itself (QUANT, lag-d and filters in one)
                extractor = SupervisedExtractor(depth = args.depth, div = args.div, num_quantiles = args.quantiles,
                                                lags = LAGS, znorm = args.znorm, catch22 = args.catch22,
                                                templates = args.templates, budget = supervised_budget(len(X_TR) + len(X_TE)),
                                                fill = args.fill_budget,
                                                c22_shorten = args.c22_shorten, rank = args.supervised_rank,
                                                max_depth = args.supervised_max_depth,
                                                quantile_fraction = args.quantile_fraction,
                                                seed = args.seed)
                t0 = time.perf_counter()
                Z_TR = extractor.fit_transform(X_TR, Y_TR).numpy()       # includes the selection
                t1 = time.perf_counter()
                Z_TE = extractor.transform(X_TE).numpy()
                t2 = time.perf_counter()
                names_Z = extractor.kept_names
                n_features_quant, n_features_lag, n_features_extra = extractor.n_candidates, 0, 0
                n_kept_quant, n_kept_lag, n_kept_extra = Z_TR.shape[1], 0, 0
                time_select = time_lag_train = time_lag_test = time_extra_train = time_extra_test = 0.0
                if args.feature_log:
                    log_selection(dataset, i, extractor.selection_log)
            else:
                quant = Quant(
                    depth = args.depth,
                    div = args.div,
                    **QUANTILES,
                )

                t0 = time.perf_counter()
                Q_TR = quant.fit_transform(X_TR, Y_TR).numpy()
                t1 = time.perf_counter()
                Q_TE = quant.transform(X_TE).numpy()
                t2 = time.perf_counter()

                n_features_quant = Q_TR.shape[1]
                names_Q = [(REPRESENTATIONS[r], kind) for r in range(len(REPRESENTATIONS)) for kind in feature_kinds(quant.models[r])]
                time_select = 0.0

                # == stage 1: QUANT + lag-d differences ==============================
                L_TR, L_TE, time_lag_train, time_lag_test, names_L = lag_features(X_TR, X_TE)
                names_S = names_Q + names_L
                n_features_lag = 0 if L_TR is None else L_TR.shape[1]
                n_kept_quant, n_kept_lag = n_features_quant, n_features_lag
                k_lag = args.lag_keep if args.lag_keep > 0 else n_features_quant
                if L_TR is None:
                    S_TR, S_TE = Q_TR, Q_TE
                elif args.prune != "none" and args.lag_filter == "separate":
                    # v3-sep: all QUANT features + the best lag-d features
                    t3 = time.perf_counter()
                    keep = best_k(L_TR, Y_TR, k_lag)
                    S_TR, S_TE = np.hstack([Q_TR, L_TR[:, keep]]), np.hstack([Q_TE, L_TE[:, keep]])
                    names_S = names_Q + [names_L[k] for k in keep]
                    time_select += time.perf_counter() - t3
                    n_kept_lag = len(keep)
                else:
                    # v3-joint: QUANT + lag-d, filtered jointly
                    S_TR, S_TE = np.hstack([Q_TR, L_TR]), np.hstack([Q_TE, L_TE])
                    if args.prune != "none":
                        t3 = time.perf_counter()
                        keep = best_k(S_TR, Y_TR, k_lag)
                        S_TR, S_TE = S_TR[:, keep], S_TE[:, keep]
                        names_S = [names_S[k] for k in keep]
                        time_select += time.perf_counter() - t3
                        n_kept_quant = int((keep < n_features_quant).sum())
                        n_kept_lag = len(keep) - n_kept_quant

                # == stage 2: + the best extra features (stage-1 features kept) =========
                if args.extra != "none":
                    model, E_TR, time_extra_train = extra_fit(X_TR, Y_TR, args.seed)
                    n_features_extra = E_TR.shape[1]
                    keep = None
                    k_extra = args.extra_keep if args.extra_keep > 0 else n_features_quant
                    if args.prune != "none":
                        t3 = time.perf_counter()
                        keep = (best_k_importance(E_TR, Y_TR, k_extra, args.seed) if args.prune == "importance"
                                else best_k(E_TR, Y_TR, k_extra))
                        E_TR = E_TR[:, keep]
                        time_select += time.perf_counter() - t3
                    # the test set: only the kept features (selected), or all features then the kept columns (all)
                    E_TE, time_extra_test, t_columns = extra_test(model, X_TE, keep)
                    time_select += t_columns
                    del model
                    n_kept_extra = E_TR.shape[1]
                    names_Z = names_S + [("extra", args.extra)] * E_TR.shape[1]
                    Z_TR = np.concatenate([S_TR, E_TR], axis = 1, dtype = np.float32)
                    Z_TE = np.concatenate([S_TE, E_TE], axis = 1, dtype = np.float32)
                    del E_TR, E_TE
                else:
                    Z_TR, Z_TE = S_TR, S_TE
                    names_Z = names_S
                    n_features_extra = n_kept_extra = 0
                    time_extra_train = time_extra_test = 0.0
                assert len(names_Z) == Z_TR.shape[1]       # one name per column
                if args.feature_log:                       # steps 1 and 2 (after the timers, so not timed)
                    log_features(dataset, i, "1 produced", names_Q + names_L + [("extra", args.extra)] * n_features_extra)
                    log_features(dataset, i, "2 kept by the filters", names_Z)
                del Q_TR, Q_TE, L_TR, L_TE, S_TR, S_TE

        except Exception as error:      # transform failed: record it for every classifier and carry on
            status = f"failed: transform: {type(error).__name__}: {error}".replace("\n", " ")[:300]
            print(f"\n{dataset} resample {i}: {status}")
            for name in todo:
                save({"dataset": dataset, "resample": i, "classifier": name, "lags": args.lags,
                      "extra": args.extra, "multirocket_kernels": args.multirocket_kernels,
                      "rdst_shapelets": args.rdst_shapelets, "prune": args.prune, "lag_filter": args.lag_filter,
                      "extra_test": args.extra_test,
                      "seed": args.seed, "status": status})
            continue

        # == each classifier on the same features ===============================

        for name in todo:

            row = {
                "dataset": dataset, "resample": i, "classifier": name, "seed": args.seed,
                "cpu_cores": "all" if args.n_jobs == -1 else args.n_jobs,
                "device": "cpu" if name == "extratrees" else args.device,
                "lags": args.lags,
                "extra": args.extra,
                "multirocket_kernels": args.multirocket_kernels if args.extra in ("mrhydra", "multirocket") else 0,
                "rdst_shapelets": args.rdst_shapelets if args.extra == "rdst" else 0,
                "prune": args.prune,
                "lag_filter": args.lag_filter,
                "extra_test": args.extra_test,
                "quant": args.quant,
                "quantiles": args.quantiles if args.quant == "tabpfn" else "",
                "supervised": SUPERVISED,
                "tabpfn_rank": args.tabpfn_rank if name == "tabpfn" else "",
                "tabpfn_estimators": (args.tabpfn_estimators or "default") if name == "tabpfn" else "",
                "tabpfn_context": args.tabpfn_context if name == "tabpfn" else "",
                "tabpfn_cache": args.tabpfn_cache if name == "tabpfn" else "",
                "n_train": len(Y_TR),
                "n_features": Z_TR.shape[1],                  # features given to the classifier
                "n_features_quant": n_features_quant,         # features computed ...
                "n_features_lag": n_features_lag,
                "n_features_extra": n_features_extra,
                "n_kept_quant": n_kept_quant,                 # ... and kept after filtering
                "n_kept_lag": n_kept_lag,
                "n_kept_extra": n_kept_extra,
                "time_transform_train": t1 - t0,              # QUANT only
                "time_transform_test": t2 - t1,
                "time_lag_train": time_lag_train,
                "time_lag_test": time_lag_test,
                "time_extra_train": time_extra_train,         # MR-Hydra, MultiRocket or RDST
                "time_extra_test": time_extra_test,
                "time_select": time_select,
            }

            try:
                if name == "tabpfn" and 0 < len(Y_TR) <= args.tabpfn_context:
                    raise SkipRun(f"{len(Y_TR)} training cases, not more than --tabpfn_context {args.tabpfn_context} "
                                  f"(the same run as without it)")

                if name == "catboost" and args.device == "cuda" and Z_TR.shape[1] > args.catboost_max_features:
                    raise SkipRun(f"{Z_TR.shape[1]} features > --catboost_max_features {args.catboost_max_features} (GPU memory)")

                if name == "xgboost" and args.device == "cuda" and Z_TR.shape[1] > args.xgboost_max_features:
                    raise SkipRun(f"{Z_TR.shape[1]} features > --xgboost_max_features {args.xgboost_max_features} (GPU memory)")

                t3 = time.perf_counter()

                # TabPFN has a feature budget, so it gets the k most important features;
                # selecting them is part of training, so it is included in the fit time
                # (no column selection for the other classifiers: Z_TR[:, all columns] would copy the
                # whole matrix, several GB on large datasets)
                # (and, with --tabpfn_context, a stratified subset of the training cases)
                if name == "tabpfn":
                    keep = select_features(Z_TR, Y_TR, args.tabpfn_features, args.seed)
                    rows = context_subset(Y_TR, args.tabpfn_context, args.seed) if args.tabpfn_context > 0 else np.arange(len(Y_TR))
                    fit_X, fit_Y, predict_X = Z_TR[rows][:, keep], Y_TR[rows], Z_TE[:, keep]
                else:
                    keep = None
                    fit_X, fit_Y, predict_X = Z_TR, Y_TR, Z_TE

                classifier = make_classifier(name, n_classes, args.seed)
                classifier.fit(fit_X, fit_Y)
                t4 = time.perf_counter()

                proba = classifier.predict_proba(predict_X)
                t5 = time.perf_counter()

                row.update(compute_metrics(Y_TE, proba, n_classes))
                row.update({
                    "n_features_used": Z_TR.shape[1] if keep is None else len(keep),
                    "n_train_used": len(fit_Y),
                    "time_fit": t4 - t3,
                    "time_predict": t5 - t4,
                    "status": "ok",
                })
                if args.feature_log and name == "tabpfn":     # step 3: the features TabPFN gets
                    log_features(dataset, i, "3 chosen by TabPFN", [names_Z[k] for k in keep])

            except SkipRun as reason:       # a known limit, e.g. too many classes or features
                row["status"] = f"skipped: {reason}"
                print(f"\n{dataset} resample {i} {name}: {row['status']}")

            except Exception as error:      # record the failure (e.g. GPU out of memory) and carry on
                row["status"] = f"failed: {type(error).__name__}: {error}".replace("\n", " ")[:300]
                print(f"\n{dataset} resample {i} {name}: {row['status']}")

            finally:
                # free the GPU memory before the next classifier: PyTorch (TabPFN) otherwise keeps
                # its memory reserved, and on Windows the next GPU classifier then spills into
                # (very slow) system memory instead of failing
                classifier = proba = fit_X = predict_X = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            save(row)       # save this run immediately