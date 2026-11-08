# A1: QUANT with a different classifier (GPU boosting or a tabular foundation model).
#
# Modified copy of the authors' original/code/run.py (GPL-3.0).
# By default (--interval_features quantile) the QUANT features are exactly the same as in
# the paper and only the classifier changes. With --classifier extratrees this script is
# the "v1" baseline: the original method, with the extra measurements added.
# quant.py can also extract summary statistics (mean, std, min, max, skew, slope, ...)
# instead of quantiles: --interval_features stats (everywhere) or auto (only at depth
# levels where quantiles would outnumber the statistics by --stats_ratio, default 2x).
#
# Changes compared with the original run.py:
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
#   .venv\Scripts\python.exe improvedQuant\v1_classifiers\run.py -i data/ts -p data/indices
#       -d improvedQuant/dataSetLists/smoke-small.txt -r 3 -c all -o results/v1_smoke-small_allcores-gpu.csv
#   ("-c all" runs extratrees, xgboost, catboost and tabpfn on the same QUANT features;
#    tabpfn needs the GPU environment .venv-gpu)

import argparse
import csv
import gc
import os
import re
import time

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

parser.add_argument("-m", "--depth", type = int, default = 6)
parser.add_argument("-v", "--div", type = int, default = 4)
parser.add_argument("--depths", default=None,
                    help="Comma-separated zero-based depths to retain, e.g. 0,2,4,6. Default: all levels below --depth.")
parser.add_argument("--interval_types", choices=["both", "normal", "shift", "alternating"], default="both",
                    help="Interval types: both, normal only, shifted only, or alternating (even depths normal, odd depths shifted).")

parser.add_argument("--interval_features", choices=["quantile", "stats", "auto"], default="quantile",
                    help="Per-interval features: quantile (original QUANT, default), stats (summary statistics everywhere), "
                         "or auto (statistics at levels where quantiles would be >= stats_ratio x the number of statistics, quantiles elsewhere).")
parser.add_argument("--stats", nargs="+", default=None,
                    choices=["mean", "std", "min", "max", "nmp", "skew", "kurt", "slope", "argmax", "argmin", "mac", "mcr"],
                    help="Statistics to extract (stats/auto only). Default: mean std min max nmp skew slope argmax argmin.")
parser.add_argument("--stats_ratio", type=float, default=2.0,
                    help="auto: use statistics at a level when its quantile count >= stats_ratio * number of statistics.")

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
# GPU memory (6 GB): CatBoost needs memory in proportion to the number of features (42,081 features asked for
# 9.2 GB), and a CUDA out-of-memory error inside CatBoost kills the whole process. XGBoost does not fail but spills
# into (very slow) system memory and hangs: at 84,162 features the laptop crashed, 48,030 and 41,978 features hung,
# although 42,081 once ran. So both skip feature sets above this safe limit
parser.add_argument("--catboost_max_features", type = int, default = 14000)
parser.add_argument("--xgboost_max_features", type = int, default = 14000)

args = parser.parse_args()

classifiers = CLASSIFIERS if "all" in args.classifier else list(dict.fromkeys(args.classifier))

_max_features = float(args.max_features) if args.max_features not in ["sqrt"] else args.max_features

# single core (-j 1), as in the paper's timings: the numerical libraries read these
# variables when they are imported, so they are set before the imports below
if args.n_jobs == 1:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
import torch

from quant import Quant

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
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
        return TabPFNClassifier(device = args.device, random_state = seed)

def select_features(Z_TR, Y_TR, k, seed):
    """Indices of the k most important QUANT features, from an ExtraTrees fit on the TRAINING data only."""
    if Z_TR.shape[1] <= k:
        return np.arange(Z_TR.shape[1])
    ranker = make_extratrees(seed).fit(Z_TR, Y_TR)
    return np.sort(np.argsort(ranker.feature_importances_)[::-1][:k])

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
    "dataset", "resample", "classifier", "seed", "cpu_cores", "device", "interval_features", "n_features", "n_features_used",
    "accuracy", "balanced_accuracy", "auroc", "log_loss",
    "time_transform_train", "time_fit", "time_transform_test", "time_predict", "status",
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
        done = {(row["dataset"], int(row["resample"]), row["classifier"]) for row in reader}
    print(f"resuming: {len(done)} runs already in {args.output_file}")
else:
    with open(args.output_file, "w", newline = "") as f:
        csv.writer(f).writerow(COLUMNS)

def save(row):
    """Append one run to the results file, in the file's own column order (also for files of an older version)."""
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

for dataset in tqdm(datasets, ncols = 80):

    for i in range(args.num_resamples):

        todo = [name for name in classifiers if (dataset, i, name) not in done]
        if not todo:
            continue

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

        # == transform (unchanged QUANT), once, shared by all classifiers ===========

        try:
            selected_depths = None if args.depths is None else [
                int(value.strip()) for value in args.depths.split(",") if value.strip()
            ]
            quant = Quant(
                depth = args.depth,
                div = args.div,
                depths = selected_depths,
                interval_types = args.interval_types,
                interval_features = args.interval_features,
                stats = args.stats,
                stats_ratio = args.stats_ratio,
            )

            t0 = time.perf_counter()
            Z_TR = quant.fit_transform(X_TR, Y_TR).numpy()
            t1 = time.perf_counter()
            Z_TE = quant.transform(X_TE).numpy()
            t2 = time.perf_counter()

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
                "interval_features": args.interval_features,
                "n_features": Z_TR.shape[1],
                "time_transform_train": t1 - t0,
                "time_transform_test": t2 - t1,
            }

            try:
                if name == "catboost" and args.device == "cuda" and Z_TR.shape[1] > args.catboost_max_features:
                    raise SkipRun(f"{Z_TR.shape[1]} features > --catboost_max_features {args.catboost_max_features} (GPU memory)")

                if name == "xgboost" and args.device == "cuda" and Z_TR.shape[1] > args.xgboost_max_features:
                    raise SkipRun(f"{Z_TR.shape[1]} features > --xgboost_max_features {args.xgboost_max_features} (GPU memory)")

                t3 = time.perf_counter()

                # TabPFN has a feature budget, so it gets the k most important features;
                # selecting them is part of training, so it is included in the fit time
                if name == "tabpfn":
                    keep = select_features(Z_TR, Y_TR, args.tabpfn_features, args.seed)
                else:
                    keep = np.arange(Z_TR.shape[1])

                classifier = make_classifier(name, n_classes, args.seed)
                classifier.fit(Z_TR[:, keep], Y_TR)
                t4 = time.perf_counter()

                proba = classifier.predict_proba(Z_TE[:, keep])
                t5 = time.perf_counter()

                row.update(compute_metrics(Y_TE, proba, n_classes))
                row.update({
                    "n_features_used": len(keep),
                    "time_fit": t4 - t3,
                    "time_predict": t5 - t4,
                    "status": "ok",
                })

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
                classifier = proba = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            save(row)       # save this run immediately