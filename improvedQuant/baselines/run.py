# Simple baselines every method should beat (lectures, weeks 6-9: "majority class, raw data + classifier, stats,
# 1NN-DTW"; 1NN-DTW and ROCKET come from the published bake-off redux results, notebook 05):
#   majority : the most frequent class of the training set (sklearn DummyClassifier)
#   raw      : ExtraTrees (200 trees) on the raw values, one feature per time point
#   catch22  : aeon's Catch22Classifier (catch22 + mean and standard deviation, then a random forest of 200 trees)
#   tabpfn   : TabPFN on the raw values, the same classifier as v4 (2 passes, the model built once, the GPU); series
#              longer than TabPFN's feature budget are subsampled by TabPFN itself (a random subset of time points per
#              pass). It shows what v4's extractor adds over giving TabPFN the series directly.
# The same 142 datasets x 30 resamples (resample 0 = the default split, the others from the resample indices), metrics
# and results-file layout as improvedQuant/features/run.py; one row per run, written as soon as the run finishes (a
# restart skips the runs already in the file).
#
# Example (from the quant-reproduction folder; .venv-gpu, which has aeon):
#   .venv-gpu\Scripts\python.exe improvedQuant\baselines\run.py -i data/ts -p data/indices
#       -d original/code/ucr142.txt -r 30 -o results/baselines_ucr142_allcores-cpu.csv
#   .venv-gpu\Scripts\python.exe improvedQuant\baselines\run.py -i data/ts -p data/indices
#       -d original/code/ucr142.txt -r 30 -c tabpfn -o results/baselines-tabpfn_ucr142_allcores-gpu.csv

import argparse
import csv
import os
import time

import numpy as np
from aeon.classification.feature_based import Catch22Classifier
from aeon.datasets import load_from_ts_file
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path")
parser.add_argument("-p", "--resample_indices_path")
parser.add_argument("-d", "--datasets")
parser.add_argument("-r", "--num_resamples", type = int, default = 30)
parser.add_argument("-c", "--classifier", nargs = "+", default = ["majority", "raw", "catch22"],
                    choices = ["majority", "raw", "catch22", "tabpfn"])
parser.add_argument("-s", "--seed", type = int, default = 0)
parser.add_argument("--device", default = "cuda", choices = ["cuda", "cpu"])     # for tabpfn
parser.add_argument("-o", "--output_file", default = "results.csv")
args = parser.parse_args()

if "tabpfn" in args.classifier:
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"          # as v4 (--tabpfn_cache): the model is built once
    from tabpfn import TabPFNClassifier
    # warm-up: CUDA and the model load here, not in the first timed run
    _rng = np.random.default_rng(0)
    TabPFNClassifier(n_estimators = 2, device = args.device, random_state = 0).fit(_rng.normal(size = (64, 8)),
                                                                                   np.arange(64) % 2)

COLUMNS = ["dataset", "resample", "classifier", "seed", "n_train", "accuracy", "balanced_accuracy", "f1_macro", "auroc",
           "log_loss", "brier", "time_fit", "time_predict", "status"]

def make_classifier(name):
    if name == "majority":
        return DummyClassifier(strategy = "most_frequent")
    if name == "raw":
        return ExtraTreesClassifier(n_estimators = 200, n_jobs = -1, random_state = args.seed)
    if name == "tabpfn":                                 # as v4: improvedQuant/features/run.py, --tabpfn_estimators 2
        return TabPFNClassifier(n_estimators = 2, device = args.device, random_state = args.seed)
    return Catch22Classifier(n_jobs = -1, random_state = args.seed)      # its default: a random forest of 200 trees

def compute_metrics(Y_TE, proba, n_classes):
    """The same metrics as improvedQuant/features/run.py."""
    predictions = proba.argmax(axis = 1)
    labels = np.arange(n_classes)
    try:
        auroc = (roc_auc_score(Y_TE, proba[:, 1]) if n_classes == 2 else
                 roc_auc_score(Y_TE, proba, multi_class = "ovr", average = "weighted", labels = labels))
    except ValueError:
        auroc = np.nan
    return {"accuracy": accuracy_score(Y_TE, predictions),
            "balanced_accuracy": balanced_accuracy_score(Y_TE, predictions),
            "f1_macro": f1_score(Y_TE, predictions, labels = labels, average = "macro", zero_division = 0),
            "auroc": auroc,
            "log_loss": log_loss(Y_TE, np.clip(proba, 1e-15, 1), labels = labels),
            "brier": float(((proba - np.eye(n_classes)[Y_TE]) ** 2).sum(axis = 1).mean())}

done = set()
if os.path.exists(args.output_file):
    with open(args.output_file, newline = "") as f:
        done = {(r["dataset"], int(r["resample"]), r["classifier"]) for r in csv.DictReader(f) if r["seed"] == str(args.seed)}
else:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok = True)
    with open(args.output_file, "w", newline = "") as f:
        csv.writer(f).writerow(COLUMNS)

def save(row):
    with open(args.output_file, "a", newline = "") as f:
        csv.DictWriter(f, COLUMNS, extrasaction = "ignore").writerow(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

for dataset in tqdm(np.atleast_1d(np.loadtxt(args.datasets, "str")), ncols = 80):
    _X_TR, _Y_TR = load_from_ts_file(f"{args.input_path}/{dataset}/{dataset}_TRAIN.ts")
    _X_TE, _Y_TE = load_from_ts_file(f"{args.input_path}/{dataset}/{dataset}_TEST.ts")
    for i in range(args.num_resamples):
        todo = [name for name in args.classifier if (dataset, i, name) not in done]
        if not todo:
            continue
        if i == 0:                                       # the default split
            X_TR, X_TE, Y_TR, Y_TE = _X_TR, _X_TE, _Y_TR, _Y_TE
        else:                                            # the published resample indices, as in run.py
            X, Y = np.concatenate((_X_TR, _X_TE)), np.concatenate((_Y_TR, _Y_TE))
            train = np.loadtxt(f"{args.resample_indices_path}/{dataset}/resample{i}Indices_TRAIN.txt", dtype = np.int32)
            test = np.setdiff1d(np.arange(len(X)), train)
            X_TR, X_TE, Y_TR, Y_TE = X[train], X[test], Y[train], Y[test]
        le = LabelEncoder()
        Y_TR, Y_TE = le.fit_transform(Y_TR), le.transform(Y_TE)
        n_classes = len(le.classes_)
        for name in todo:
            row = {"dataset": dataset, "resample": i, "classifier": name, "seed": args.seed, "n_train": len(Y_TR)}
            try:
                classifier = make_classifier(name)
                t0 = time.perf_counter()
                # raw values: one column per time point (the univariate series as a table)
                table = name in ("raw", "tabpfn")
                classifier.fit(X_TR[:, 0, :] if table else X_TR, Y_TR)
                t1 = time.perf_counter()
                proba = np.zeros((len(Y_TE), n_classes))
                proba[:, classifier.classes_] = classifier.predict_proba(X_TE[:, 0, :] if table else X_TE)
                t2 = time.perf_counter()
                row.update(compute_metrics(Y_TE, proba, n_classes), time_fit = t1 - t0, time_predict = t2 - t1, status = "ok")
            except Exception as error:
                row["status"] = f"failed: {type(error).__name__}: {error}".replace("\n", " ")[:300]
                print(f"\n{dataset} resample {i} {name}: {row['status']}")
            save(row)
