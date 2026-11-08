import argparse
import csv
import gc
import os
import re
import time
from collections import Counter

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

parser.add_argument("-m", "--depth", type = int, default = 6)               # QUANT: levels 0 to depth - 1
parser.add_argument("-v", "--div", type = int, default = 4)                 # QUANT: one quantile per div points

parser.add_argument("-e", "--num_estimators", type = int, default = 200)   # all three classifiers
parser.add_argument("-f", "--max_features", type = float, default = 0.1)   # all three classifiers

CLASSIFIERS = ["extratrees", "extratrees-k", "extratrees-dup"]
parser.add_argument("-c", "--classifier", nargs = "+", default = CLASSIFIERS, choices = CLASSIFIERS + ["all"])
parser.add_argument("--weight_power", type = float, default = 1.0)        # the copies, see the top of the file
parser.add_argument("-s", "--seed", type = int, default = 0)
parser.add_argument("-j", "--n_jobs", type = int, default = -1)             # 1 = single CPU core

args = parser.parse_args()

classifiers = CLASSIFIERS if "all" in args.classifier else list(dict.fromkeys(args.classifier))

# single core (-j 1), as in the paper's timings: the numerical libraries read these
# variables when they are imported, so they are set before the imports below
if args.n_jobs == 1:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
import torch

from quant import Quant, make_intervals

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

if args.n_jobs == 1:
    torch.set_num_threads(1)            # the QUANT transform on one core
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
# == the level of every QUANT feature ==========================================
# ==============================================================================

def interval_levels(input_length, depth):
    """The level of every interval of quant.make_intervals(input_length, depth), in its order: the same loop as
    make_intervals, with the level of each interval (a shifted copy has the level it is shifted from)."""
    exponent = min(depth, int(np.log2(input_length)) + 1)
    levels = []
    for level, n in enumerate(2 ** torch.arange(exponent)):
        indices = torch.linspace(0, input_length, n + 1).long()
        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)
        levels += [level] * len(intervals_n)
        if n > 1 and intervals_n.diff().median() > 1:
            levels += [level] * (len(intervals_n) - 1)
    return levels

def feature_levels(quant, depth, div):
    """(view, level) of every column of a fitted Quant's features, in the column order. A piece of m > 1 points gives
    1 + (m - 1) // div quantiles, a piece of 1 point 1 value (quant.f_quantile)."""
    columns = []
    for view, model in quant.models.items():
        input_length = int(model.intervals[:, 1].max())        # the last interval ends at the view's length
        assert torch.equal(make_intervals(input_length, depth), model.intervals)
        levels = interval_levels(input_length, depth)
        assert len(levels) == len(model.intervals)
        for (a, b), level in zip(model.intervals.tolist(), levels):
            m = b - a
            columns += [(view, level)] * (1 if m == 1 else 1 + (m - 1) // div)
    return columns

def column_copies(columns, power):
    """extratrees-dup: how many times each column is given to ExtraTrees = round((features of the largest level of its
    view / features of its own level) ^ power), at least 1. The levels are compared within each view, never across
    views, so only the larger intervals are copied (a view with fewer features, e.g. |FFT|, is not copied as a whole).
    power 1: every level of a view drawn about equally often."""
    counts = Counter(columns)                                   # (view, level) -> number of features
    largest = {}
    for (view, level), n in counts.items():
        largest[view] = max(largest.get(view, 0), n)
    return np.array([max(1, round((largest[c[0]] / counts[c]) ** power)) for c in columns], dtype = np.int64)

# ==============================================================================
# == classifiers ===============================================================
# ==============================================================================

def candidates_per_split(name, n_quant, n_dup):
    """ExtraTrees' candidate features per split: 10% of its columns (as scikit-learn computes it for a fraction);
    extratrees-k: as many as extratrees-dup, from QUANT's own columns (at most all of them)."""
    k_dup = max(1, int(args.max_features * n_dup))
    return {"extratrees": max(1, int(args.max_features * n_quant)), "extratrees-k": min(k_dup, n_quant),
            "extratrees-dup": k_dup}[name]

def make_classifier(name, n_quant, n_dup, seed):
    """scikit-learn's ExtraTrees with QUANT's settings (200 trees, entropy); extratrees-k gets its number of candidates
    per split as a count, the other two as the fraction 0.1 of their columns."""
    max_features = candidates_per_split(name, n_quant, n_dup) if name == "extratrees-k" else args.max_features
    return ExtraTreesClassifier(n_estimators = args.num_estimators, max_features = max_features,
                                criterion = "entropy", n_jobs = args.n_jobs, random_state = seed)

def compute_metrics(Y_TE, proba, n_classes):
    """Accuracy, balanced accuracy, AUROC and log loss from the class probabilities (as v1)."""
    predictions = proba.argmax(axis = 1)
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
    "dataset", "resample", "classifier", "depth", "div", "weight_power", "seed", "cpu_cores", "n_features", "n_levels",
    "n_columns", "candidates_per_split", "accuracy", "balanced_accuracy", "auroc", "log_loss",
    "time_transform_train", "time_fit", "time_transform_test", "time_predict", "status",
]

datasets = np.atleast_1d(np.loadtxt(f"{args.datasets}", "str"))   # atleast_1d: a list with one name works too

# results file: create it with a header, or resume from the runs already in it
os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok = True)
done = set()
if os.path.exists(args.output_file):
    with open(args.output_file, newline = "") as f:
        done = {(row["dataset"], int(row["resample"]), row["classifier"]) for row in csv.DictReader(f)}
    print(f"resuming: {len(done)} runs already in {args.output_file}")
else:
    with open(args.output_file, "w", newline = "") as f:
        csv.writer(f).writerow(COLUMNS)

def save(row):
    """Append one run to the results file."""
    with open(args.output_file, "a", newline = "") as f:
        csv.DictWriter(f, COLUMNS, extrasaction = "ignore").writerow(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

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
            quant = Quant(depth = args.depth, div = args.div)

            t0 = time.perf_counter()
            Z_TR = quant.fit_transform(X_TR, Y_TR).numpy()
            t1 = time.perf_counter()
            Z_TE = quant.transform(X_TE).numpy()
            t2 = time.perf_counter()

            columns = feature_levels(quant, args.depth, args.div)     # not timed: the same for every classifier
            assert len(columns) == Z_TR.shape[1], (len(columns), Z_TR.shape)
            copies = column_copies(columns, args.weight_power)
            n_quant, n_dup = Z_TR.shape[1], int(copies.sum())

        except Exception as error:      # transform failed: record it for every classifier and carry on
            status = f"failed: transform: {type(error).__name__}: {error}".replace("\n", " ")[:300]
            print(f"\n{dataset} resample {i}: {status}")
            for name in todo:
                save({"dataset": dataset, "resample": i, "classifier": name, "seed": args.seed, "status": status})
            continue

        # == each classifier on the same features ===============================

        for name in todo:

            row = {
                "dataset": dataset, "resample": i, "classifier": name, "depth": args.depth, "div": args.div,
                "weight_power": args.weight_power if name != "extratrees" else "",
                "seed": args.seed, "cpu_cores": "all" if args.n_jobs == -1 else args.n_jobs,
                "n_features": n_quant, "n_levels": len({level for _, level in columns}),
                "n_columns": n_dup if name == "extratrees-dup" else n_quant,      # the columns ExtraTrees gets
                "candidates_per_split": candidates_per_split(name, n_quant, n_dup),
                "time_transform_train": t1 - t0,
                "time_transform_test": t2 - t1,
            }

            try:
                t3 = time.perf_counter()
                # extratrees-dup: the copies are made here, so they count in the fit time (test data: predict time)
                fit_X = np.repeat(Z_TR, copies, axis = 1) if name == "extratrees-dup" else Z_TR
                classifier = make_classifier(name, n_quant, n_dup, args.seed)
                classifier.fit(fit_X, Y_TR)
                t4 = time.perf_counter()

                predict_X = np.repeat(Z_TE, copies, axis = 1) if name == "extratrees-dup" else Z_TE
                proba = classifier.predict_proba(predict_X)
                t5 = time.perf_counter()

                row.update(compute_metrics(Y_TE, proba, n_classes))
                row.update({"time_fit": t4 - t3, "time_predict": t5 - t4, "status": "ok"})

            except Exception as error:      # record the failure (e.g. out of memory) and carry on
                row["status"] = f"failed: {type(error).__name__}: {error}".replace("\n", " ")[:300]
                print(f"\n{dataset} resample {i} {name}: {row['status']}")

            finally:
                # the copied matrices can be several GB on large datasets at depth 8
                classifier = proba = fit_X = predict_X = None
                gc.collect()

            save(row)       # save this run immediately
