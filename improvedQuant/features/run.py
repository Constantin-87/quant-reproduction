# The improvements after v1 (v2, v3 and v4; experiments.md lists every version with its command), with ExtraTrees or
# TabPFN as the classifier. One script for all of them; the version is chosen by the options:
#   v2: TabPFN's cost settings on QUANT's features (--quant original, the default): --tabpfn_estimators (passes),
#       --tabpfn_features (the k most important features, ranked on the training data) and --tabpfn_cache (TabPFN's
#       model built once per process);
#   v3: the lean extractor quant_tabpfn.py (--quant tabpfn) + lag-d differences x(t) - x(t-d), d in --lags, summarised
#       with the same interval features; all QUANT features are kept and the best lag-d features are added (ANOVA
#       F-score on the training data, as many as QUANT has);
#   v4: the supervised extractor quant_tabpfn.SupervisedExtractor (--quant supervised), which picks its features
#       itself; the ablation of notebook 04 switches its parts off one at a time.
#
# Built on improvedQuant/v1_classifiers/run.py, a modified copy of the authors' original/code/run.py (GPL-3.0);
# the comparison with XGBoost and CatBoost (v1) is in that script. quant.py is the authors' file, unchanged;
# quant_tabpfn.py is a changed copy for TabPFN (its header lists the changes).
#
# Changes compared with the original run.py:
#   - --classifier: extratrees (original) or tabpfn
#   - fixed random seed (--seed); the original run.py is unseeded
#   - metrics from predict_proba: accuracy, balanced accuracy, AUROC, log loss
#   - per-phase timers: transform train, fit, transform test, predict
#   - one CSV row written after every run (a crash does not lose earlier results;
#     a restart skips runs that are already in the file)
#   - a failed run (e.g. GPU out of memory) is recorded with its error instead of
#     stopping the whole experiment
#   - also macro F1, the Brier score and the peak memory of every run (RAM sampled during the run, GPU from PyTorch);
#     --save_predictions writes every run's test labels and class probabilities (confusion matrices, notebook 05)
#   - runs with different seeds (-s) can share one results file (a restart skips the runs of the same seed)
#
# Example (from the quant-reproduction folder; .venv-gpu: TabPFN needs CUDA PyTorch, catch22 comes from aeon):
#   .venv-gpu\Scripts\python.exe improvedQuant\features\run.py -i data/ts -p data/indices
#       -d improvedQuant/dataSetLists/smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200
#       -o results/v2_tabpfn-e2f200_smoke-small_allcores-gpu.csv

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

# the depth (levels of pieces; default: 6 original, 5 tabpfn, 4 supervised) and, for the authors' quant.py only, div
# (one quantile per div points; default 4)
parser.add_argument("-m", "--depth", type = int, default = None)
parser.add_argument("-v", "--div", type = int, default = 4)
# the feature extractor: "original" = the authors' quant.py, "tabpfn" = quant_tabpfn.py (fewer, less redundant
# features for TabPFN's small feature budget), "supervised" = quant_tabpfn.SupervisedExtractor
parser.add_argument("--quant", default = "original", choices = ["original", "tabpfn", "supervised"])
# --quant tabpfn / supervised: a fixed number of quantiles per piece (2 = min and max)
parser.add_argument("--quantiles", type = int, default = 2)
# lag-d differences: e.g. "4,16" (--quant original / tabpfn: extra views, the best ones kept); "auto" (--quant
# supervised: a ladder 2, 4, 8, ... the extractor chooses from per dataset); "none" = no lag-d differences
parser.add_argument("--lags", default = "none")

# --quant supervised: the extractor returns at most --supervised_budget features, chosen on the training labels
# (relevance check + greedy selection without near-copies); its parts:
parser.add_argument("--supervised_budget", type = int, default = 250)
parser.add_argument("--catch22", action = "store_true")              # + catch22 of the series and of its change
parser.add_argument("--templates", action = "store_true")            # + pattern templates
# fill the budget: past p >= 0.05, then finer pieces down to --supervised_max_depth
parser.add_argument("--fill_budget", action = "store_true")
# the deepest pieces of the fill rounds: 5 = down to 1/16 of the series (v4)
parser.add_argument("--supervised_max_depth", type = int, default = 6)
# catch22 on a shortened copy (the mean of every k points), k chosen per dataset from the training series
parser.add_argument("--c22_shorten", action = "store_true")
# the order of the greedy selection: anova = the ANOVA ranking; both = ANOVA keeps the budget + half of the rest,
# then the ExtraTrees importance ranks them (v4)
parser.add_argument("--supervised_rank", default = "anova", choices = ["anova", "both"])
# the quantiles of QUANT's four views: 0 = min and max; 0.2 = a fifth of QUANT's (one per 4 points), odd so that the
# median stays in (v4; the lag-d views keep min and max)
parser.add_argument("--quantile_fraction", type = float, default = 0)
# also write <output>_selection.csv: what happened to every candidate of the supervised extractor (notebook 04)
parser.add_argument("--feature_log", action = "store_true")

parser.add_argument("-e", "--num_estimators", type = int, default = 200)   # ExtraTrees
parser.add_argument("-f", "--max_features", default = "0.1")               # ExtraTrees

CLASSIFIERS = ["extratrees", "tabpfn"]
# one or more classifiers, e.g. "-c extratrees tabpfn"; they share the same features
parser.add_argument("-c", "--classifier", nargs = "+", default = ["extratrees"], choices = CLASSIFIERS)
parser.add_argument("-s", "--seed", type = int, default = 0)
parser.add_argument("-j", "--n_jobs", type = int, default = -1)             # 1 = single CPU core
parser.add_argument("--device", default = "cuda", choices = ["cuda", "cpu"])  # for tabpfn
parser.add_argument("--tabpfn_features", type = int, default = 500)        # TabPFN: keep only the k most important features
# how TabPFN's features are picked: "extratrees" (the most important in an ExtraTrees fit) or "both" (ANOVA drops the
# worse half beyond k, then ExtraTrees picks the k most important)
parser.add_argument("--tabpfn_rank", default = "extratrees", choices = ["extratrees", "both"])
# TabPFN's ensemble size (number of forward passes that are averaged); 0 = TabPFN's default (8)
parser.add_argument("--tabpfn_estimators", type = int, default = 0)
# TabPFN's own model cache: build the model once per process instead of for every run (about 2 s per run, the
# same predictions)
parser.add_argument("--tabpfn_cache", action = "store_true")
# also write every run's test labels and class probabilities to <output>_predictions/<dataset>_<resample>_<classifier>_
# s<seed>.npz (for confusion matrices and metrics computed later)
parser.add_argument("--save_predictions", action = "store_true")

args = parser.parse_args()

classifiers = list(dict.fromkeys(args.classifier))

if args.depth is None:
    args.depth = {"original": 6, "tabpfn": 5, "supervised": 4}[args.quant]

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
# how many quantiles per piece: the authors' quant.py one per div points, quant_tabpfn.py a fixed number
QUANTILES = {"div": args.div} if args.quant == "original" else {"num_quantiles": args.quantiles}
# the settings of the supervised extractor, as one short text for the results file, e.g. "c1-t1-b200-f1-s1-kboth-d5-q0.2"
SUPERVISED = (f"c{int(args.catch22)}-t{int(args.templates)}-b{args.supervised_budget}-f{int(args.fill_budget)}"
              f"-s{int(args.c22_shorten)}-k{args.supervised_rank}-d{args.supervised_max_depth}"
              + (f"-q{args.quantile_fraction:g}" if args.quantile_fraction > 0 else "")
              if args.quant == "supervised" else "")
assert args.quantile_fraction == 0 or args.quant == "supervised", "--quantile_fraction needs --quant supervised"

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
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
# == classifiers ===============================================================
# ==============================================================================

def make_extratrees(seed):
    # the original QUANT classifier (paper, Section 3.4), now with a seed
    return ExtraTreesClassifier(
        n_estimators = args.num_estimators,
        max_features = _max_features,
        criterion = "entropy",
        n_jobs = args.n_jobs,
        random_state = seed,
    )

def make_classifier(name, seed):
    """Return a new classifier. Settings are fixed in advance (library defaults, TabPFN's passes)."""
    if name == "extratrees":
        return make_extratrees(seed)
    from tabpfn import TabPFNClassifier
    if args.tabpfn_estimators > 0:
        return TabPFNClassifier(n_estimators = args.tabpfn_estimators, device = args.device, random_state = seed)
    return TabPFNClassifier(device = args.device, random_state = seed)

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

def select_features(Z_TR, Y_TR, k, seed):
    """Indices of the k best features for TabPFN, on the TRAINING data only (--tabpfn_rank):
    extratrees = the most important in an ExtraTrees fit, both = ANOVA drops the worse half of the features beyond k,
    then the k most important of the rest in an ExtraTrees fit (e.g. 450 features, k = 200: ANOVA keeps 325,
    ExtraTrees picks 200)."""
    n = Z_TR.shape[1]
    if n <= k:
        return np.arange(n)
    candidates = best_k(Z_TR, Y_TR, k + (n - k) // 2) if args.tabpfn_rank == "both" else np.arange(n)
    ranker = make_extratrees(seed).fit(Z_TR[:, candidates], Y_TR)
    return np.sort(candidates[np.argsort(ranker.feature_importances_)[::-1][:k]])

def compute_metrics(Y_TE, proba, n_classes):
    """Accuracy, balanced accuracy, AUROC and log loss from the class probabilities."""
    predictions = proba.argmax(axis = 1)    # same as predict() for both classifiers
    labels = np.arange(n_classes)
    try:
        if n_classes == 2:
            auroc = roc_auc_score(Y_TE, proba[:, 1])
        else:
            auroc = roc_auc_score(Y_TE, proba, multi_class = "ovr", average = "weighted", labels = labels)
    except ValueError:              # e.g. a class that does not occur in this test set
        auroc = np.nan
    onehot = np.eye(n_classes)[Y_TE]
    return {
        "accuracy": accuracy_score(Y_TE, predictions),
        "balanced_accuracy": balanced_accuracy_score(Y_TE, predictions),
        "f1_macro": f1_score(Y_TE, predictions, labels = labels, average = "macro", zero_division = 0),
        "auroc": auroc,
        "log_loss": log_loss(Y_TE, np.clip(proba, 1e-15, 1), labels = labels),
        "brier": float(((proba - onehot) ** 2).sum(axis = 1).mean()),    # multi-class Brier score (0 = perfect)
    }

# ==============================================================================
# == memory ====================================================================
# ==============================================================================

def process_memory_mb():
    """The memory this process uses now (MB): its private bytes on Windows, its resident set elsewhere."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(name, ctypes.c_size_t) for name in (
                           "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                           "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PagefileUsage / 2 ** 20
    with open("/proc/self/statm") as f:                   # Linux: resident pages
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20

class MemorySampler:
    """The peak memory of the process during a run: a background thread samples it every 20 ms (the operating system
    only keeps the peak of the whole process, which cannot be reset between runs)."""
    def __init__(self):
        import threading
        self.peak, self.running = 0.0, True
        threading.Thread(target = self._sample, daemon = True).start()

    def _sample(self):
        while self.running:
            self.peak = max(self.peak, process_memory_mb())
            time.sleep(0.02)

    def reset(self):
        self.peak = process_memory_mb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def read(self):
        """(peak RAM, peak GPU memory) since the last reset, in MB."""
        self.peak = max(self.peak, process_memory_mb())
        gpu = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
        return self.peak, gpu

# ==============================================================================
# == features ==================================================================
# ==============================================================================

# --lags auto (--quant supervised only): a ladder of lags, the extractor picks per dataset (SupervisedExtractor.lag_list)
LAGS = [] if args.lags.lower() == "none" else "auto" if args.lags == "auto" else [int(d) for d in args.lags.split(",")]
assert LAGS != "auto" or args.quant == "supervised", "--lags auto needs --quant supervised"
assert args.quant != "supervised" or LAGS in ("auto", []), "--quant supervised: --lags auto or none"

def lag_features(X_TR, X_TE):
    """v3: the interval features of the lag-d differences x(t) - x(t-d), d in --lags, with the same IntervalModel as
    QUANT's views (so each lag-d difference is one more view). Lags too long for the series are skipped.
    Returns the train and test features and the transform times."""
    F_TR, F_TE, t_train, t_test = [], [], 0.0, 0.0
    for d in LAGS:
        if X_TR.shape[-1] <= d + 1:
            continue
        t0 = time.perf_counter()
        D_TR = X_TR[..., d:] - X_TR[..., :-d]
        model = IntervalModel(input_length = D_TR.shape[-1], depth = args.depth, **QUANTILES)
        F_TR.append(model.fit_transform(D_TR, None).numpy())
        t1 = time.perf_counter()
        F_TE.append(model.transform(X_TE[..., d:] - X_TE[..., :-d]).numpy())
        t_train, t_test = t_train + (t1 - t0), t_test + (time.perf_counter() - t1)
    if not F_TR:
        return None, None, 0.0, 0.0
    return np.hstack(F_TR), np.hstack(F_TE), t_train, t_test

def log_selection(dataset, i, log):
    """--quant supervised: what happened to every candidate, counted per (source, kind, level, position, outcome,
    copy_of), appended to <output>_selection.csv (notebook 04)."""
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

def features(X_TR, Y_TR, X_TE, dataset, i):
    """The training and test features of one run (the selection on the training data only), their counts and times."""
    info = {"n_features_lag": 0, "n_kept_lag": 0, "time_lag_train": 0.0, "time_lag_test": 0.0, "time_select": 0.0}
    if args.quant == "supervised":
        # the extractor returns the chosen features itself (QUANT's views, lag-d views and the selection in one)
        extractor = SupervisedExtractor(depth = args.depth, num_quantiles = args.quantiles,
                                        lags = LAGS == "auto", catch22 = args.catch22, templates = args.templates,
                                        budget = args.supervised_budget, fill = args.fill_budget,
                                        c22_shorten = args.c22_shorten, rank = args.supervised_rank,
                                        max_depth = args.supervised_max_depth,
                                        quantile_fraction = args.quantile_fraction, seed = args.seed)
        t0 = time.perf_counter()
        Z_TR = extractor.fit_transform(X_TR, Y_TR).numpy()               # includes the selection
        t1 = time.perf_counter()
        Z_TE = extractor.transform(X_TE).numpy()
        t2 = time.perf_counter()
        if args.feature_log:
            log_selection(dataset, i, extractor.selection_log)
        info.update(n_features_quant = extractor.n_candidates, n_kept_quant = Z_TR.shape[1])
        return Z_TR, Z_TE, dict(info, time_transform_train = t1 - t0, time_transform_test = t2 - t1)

    quant = Quant(depth = args.depth, **QUANTILES)
    t0 = time.perf_counter()
    Q_TR = quant.fit_transform(X_TR, Y_TR).numpy()
    t1 = time.perf_counter()
    Q_TE = quant.transform(X_TE).numpy()
    t2 = time.perf_counter()
    info.update(n_features_quant = Q_TR.shape[1], n_kept_quant = Q_TR.shape[1],
                time_transform_train = t1 - t0, time_transform_test = t2 - t1)
    L_TR, L_TE, info["time_lag_train"], info["time_lag_test"] = lag_features(X_TR, X_TE)
    if L_TR is None:
        return Q_TR, Q_TE, info
    # all QUANT features + the best lag-d features (as many as QUANT has)
    t3 = time.perf_counter()
    keep = best_k(L_TR, Y_TR, Q_TR.shape[1])
    info["time_select"] = time.perf_counter() - t3
    info.update(n_features_lag = L_TR.shape[1], n_kept_lag = len(keep))
    return np.hstack([Q_TR, L_TR[:, keep]]), np.hstack([Q_TE, L_TE[:, keep]]), info

# ==============================================================================
# == run =======================================================================
# ==============================================================================

COLUMNS = [
    "dataset", "resample", "classifier", "lags", "quant", "quantiles", "supervised", "tabpfn_rank",
    "tabpfn_estimators", "tabpfn_cache", "seed", "cpu_cores", "device",
    "n_features", "n_features_quant", "n_features_lag", "n_kept_quant", "n_kept_lag", "n_features_used", "n_train",
    "accuracy", "balanced_accuracy", "f1_macro", "auroc", "log_loss", "brier",
    "time_transform_train", "time_lag_train", "time_select", "time_fit",
    "time_transform_test", "time_lag_test", "time_predict", "peak_ram_mb", "peak_gpu_mb", "status",
]

datasets = np.atleast_1d(np.loadtxt(f"{args.datasets}", "str"))   # atleast_1d: a list with one name works too

# results file: create it with a header, or resume from the runs already in it
os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok = True)
done = set()
fieldnames = COLUMNS
if os.path.exists(args.output_file):
    with open(args.output_file, newline = "") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or COLUMNS      # a file made by an older version keeps its own columns
        # a run = dataset, resample, classifier and seed (runs with other seeds can be in the same file)
        done = {(row["dataset"], int(row["resample"]), row["classifier"]) for row in reader
                if row.get("seed", "") in ("", str(args.seed))}
    print(f"resuming: {len(done)} runs already in {args.output_file}")
else:
    with open(args.output_file, "w", newline = "") as f:
        csv.writer(f).writerow(COLUMNS)

def save(row):
    """Append one run to the results file, in the file's own column order."""
    with open(args.output_file, "a", newline = "") as f:
        csv.DictWriter(f, fieldnames, extrasaction = "ignore").writerow(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

# GPU warm-up: the first GPU fit also starts CUDA, which would inflate the first timing
if "tabpfn" in classifiers and args.device == "cuda":
    try:
        rng = np.random.default_rng(0)
        make_classifier("tabpfn", args.seed).fit(rng.normal(size = (64, 8)), np.arange(64) % 2)
    except Exception as error:
        print(f"GPU warm-up failed for tabpfn: {type(error).__name__}: {error}")
# the same for catch22 in the supervised extractor: aeon compiles it (numba) on its first call
if args.quant == "supervised" and args.catch22:
    rng = np.random.default_rng(0)
    _warm = SupervisedExtractor(catch22 = True, budget = 5)
    _warm.fit_transform(torch.tensor(rng.normal(size = (10, 1, 64)), dtype = torch.float32), np.arange(10) % 2)
    del _warm

Z_TR = Z_TE = None
memory = MemorySampler()
PREDICTIONS_DIR = args.output_file[:-4] + "_predictions"
if args.save_predictions:
    os.makedirs(PREDICTIONS_DIR, exist_ok = True)

for dataset in tqdm(datasets, ncols = 80):

    for i in range(args.num_resamples):

        todo = [name for name in classifiers if (dataset, i, name) not in done]
        if not todo:
            continue

        # free the previous run's feature matrices before building new ones
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

        # == features, once, shared by the classifiers =========================

        memory.reset()                  # the peak memory of this run: from here to the end of each classifier
        try:
            Z_TR, Z_TE, info = features(X_TR, Y_TR, X_TE, dataset, i)
        except Exception as error:      # transform failed: record it for every classifier and carry on
            status = f"failed: transform: {type(error).__name__}: {error}".replace("\n", " ")[:300]
            print(f"\n{dataset} resample {i}: {status}")
            for name in todo:
                save({"dataset": dataset, "resample": i, "classifier": name, "seed": args.seed, "status": status})
            continue

        # == each classifier on the same features ===============================

        for name in todo:

            row = {
                "dataset": dataset, "resample": i, "classifier": name, "seed": args.seed,
                "cpu_cores": "all" if args.n_jobs == -1 else args.n_jobs,
                "device": "cpu" if name == "extratrees" else args.device,
                "lags": args.lags,
                "quant": args.quant,
                "quantiles": args.quantiles if args.quant == "tabpfn" else "",
                "supervised": SUPERVISED,
                "tabpfn_rank": args.tabpfn_rank if name == "tabpfn" else "",
                "tabpfn_estimators": (args.tabpfn_estimators or "default") if name == "tabpfn" else "",
                "tabpfn_cache": args.tabpfn_cache if name == "tabpfn" else "",
                "n_train": len(Y_TR),
                "n_features": Z_TR.shape[1],                  # features given to the classifier (before TabPFN's budget)
                **info,
            }

            try:
                t3 = time.perf_counter()

                # TabPFN has a feature budget, so it gets the k most important features; selecting them is part of
                # training, so it is included in the fit time
                if name == "tabpfn":
                    keep = select_features(Z_TR, Y_TR, args.tabpfn_features, args.seed)
                    fit_X, predict_X = Z_TR[:, keep], Z_TE[:, keep]
                else:
                    keep = None
                    fit_X, predict_X = Z_TR, Z_TE

                classifier = make_classifier(name, args.seed)
                classifier.fit(fit_X, Y_TR)
                t4 = time.perf_counter()

                proba = classifier.predict_proba(predict_X)
                t5 = time.perf_counter()

                row.update(compute_metrics(Y_TE, proba, n_classes))
                peak_ram, peak_gpu = memory.read()
                row.update({
                    "n_features_used": Z_TR.shape[1] if keep is None else len(keep),
                    "time_fit": t4 - t3,
                    "time_predict": t5 - t4,
                    "peak_ram_mb": peak_ram,
                    "peak_gpu_mb": peak_gpu,
                    "status": "ok",
                })
                if args.save_predictions:
                    np.savez_compressed(f"{PREDICTIONS_DIR}/{dataset}_{i}_{name}_s{args.seed}.npz", y_test = Y_TE,
                                        proba = proba.astype(np.float32), classes = le.classes_.astype(str))

            except Exception as error:      # record the failure (e.g. GPU out of memory) and carry on
                row["status"] = f"failed: {type(error).__name__}: {error}".replace("\n", " ")[:300]
                print(f"\n{dataset} resample {i} {name}: {row['status']}")

            finally:
                # free the GPU memory before the next classifier
                classifier = proba = fit_X = predict_X = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            save(row)       # save this run immediately
