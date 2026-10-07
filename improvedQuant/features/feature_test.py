"""The tests behind v9 (notebook 05, section 12; experiments.md, v9): which of QUANT's interval features help TabPFN?

QUANT has four kinds of interval features that the supervised extractor (v8) does not: pieces down to 1/32 of the
series, shifted pieces (a half-offset copy of every level), dense quantiles (one per 4 points, the median and the inner
quantiles among them) and quantiles minus the piece's mean. This script switches them on and off, one at a time:
  Q-variants: TabPFN (2 passes) on QUANT's own features with one kind removed or changed; TabPFN gets 250 features
              picked as in v4 (run.py --tabpfn_rank both), or all of them (QA);
  V-variants: the supervised extractor with v9's settings (fill rounds down to 1/16, 250 features) with one kind added
              to the candidates of QUANT's four views (the lag-d views keep min and max), then its own selection.
Every variant writes one row per run: accuracy, features, time, and what the selection kept, by kind.

Groups of variants (--variants): ablation (QA, Q0-Q5, V0-V4), quantiles (V0, V3), anova (A0, A3: the ANOVA order
instead of the ExtraTrees importance), quant200 (QB: QUANT + TabPFN, 2 passes x 200 features picked as v4).

  .venv-gpu\\Scripts\\python.exe improvedQuant\\features\\feature_test.py -i data/ts -p data/indices
      -d improvedQuant/dataSetLists/worst7.txt -r 10 --variants ablation -o results/<file>.csv
  .venv-gpu\\Scripts\\python.exe improvedQuant\\features\\feature_test.py --check     (the features vs quant.py, quant_tabpfn.py)

Every run is written as soon as it finishes; running the same command again skips the runs already in the file.
"""
import argparse
import importlib.util
import os
import re
import sys
import time
import types
import warnings
from pathlib import Path

os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"                     # TabPFN's model built once (run.py --tabpfn_cache)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                                        # quant-reproduction/
sys.path.insert(0, str(HERE))
import quant_tabpfn as lean                                      # noqa: E402

_spec = importlib.util.spec_from_file_location("quant_orig", REPO / "original" / "code" / "quant.py")
orig = importlib.util.module_from_spec(_spec)                    # the authors' quant.py, for --check
_spec.loader.exec_module(orig)

# run.py's .ts loader and its feature picking for TabPFN, taken from run.py itself so that both are exactly the same
_src = (HERE / "run.py").read_text(encoding="utf-8").splitlines()
_line = lambda start: next(i for i, l in enumerate(_src) if l.startswith(start))
_ns = {"np": np, "pd": pd, "re": re, "os": os, "warnings": warnings, "ExtraTreesClassifier": ExtraTreesClassifier,
       "args": types.SimpleNamespace(num_estimators=200, n_jobs=-1, tabpfn_rank="both"), "_max_features": 0.1}
exec("\n".join(_src[_line("def _load_header_info"):_line("def context_subset")]
               + _src[_line("def best_k("):_line("datasets = ")]), _ns)
load_from_tsfile, select_features = _ns["load_from_tsfile"], _ns["select_features"]

QUANT_VIEWS = ["series", "change (smoothed)", "change of change", "rhythm (|FFT|)"]
REPS = [lambda X: X,                                              # QUANT's four views, as quant.py
        lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
        lambda X: X.diff(n=2),
        lambda X: torch.fft.rfft(X).abs()]


# == interval features with every option of quant.py and quant_tabpfn.py =================================================
def make_intervals(length, depth, shifted):
    """The pieces of quant.py (shifted = True) or quant_tabpfn.py (False): (start, end, pieces per level, shifted?)."""
    exponent = min(depth, int(np.log2(length)) + 1)
    out = []
    for n in 2 ** torch.arange(exponent):
        idx = torch.linspace(0, length, n + 1).long()
        iv = torch.stack((idx[:-1], idx[1:]), 1)
        out += [(int(a), int(b), int(n), False) for a, b in iv]
        if shifted and n > 1 and iv.diff().median() > 1:
            shift = int(np.ceil(length / n / 2))
            out += [(int(a), int(b), int(n), True) for a, b in (iv[:-1] + shift)]
    return out


def n_quantiles(n, scheme):
    """Quantiles of a piece of n points: "quant" = one per 4 points (quant.py); "q20" = a fifth of that, odd so the
    median stays in (quant_tabpfn.py change 8, v9); "minmax" = min and max (v4 to v8)."""
    if n == 1:
        return 1
    if scheme == "quant":
        return 1 + (n - 1) // 4
    if scheme == "q20":
        k = max(1, round((1 + (n - 1) // 4) * 0.2))
        return min(k + 1 - k % 2, n)
    return min(2, n)


def quantile_kinds(k):
    if k == 1:
        return ["median"]
    return ["min" if x == 0 else "max" if x == 1 else "median" if abs(x - 0.5) < 1e-9 else "inner"
            for x in np.linspace(0, 1, k)]


class GenIntervals:
    """The interval features of one view, with one switch per option."""

    def __init__(self, length, depth, shifted, scheme, mean_sub, summaries):
        self.intervals = make_intervals(length, depth, shifted)
        self.scheme, self.mean_sub, self.summaries = scheme, mean_sub, summaries

    def transform(self, X):
        feats = []
        for a, b, _, _ in self.intervals:
            Z = X[..., a:b]
            n = Z.shape[-1]
            k = n_quantiles(n, self.scheme)
            if n == 1:
                q = Z.reshape(Z.shape[0], -1)
            elif k == 1:
                q = Z.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0).reshape(Z.shape[0], -1)
            else:
                q = Z.quantile(torch.linspace(0, 1, k), dim=-1).permute(1, 2, 0)
                if self.mean_sub:                                # quant.py: every second quantile minus the mean
                    q[..., 1::2] = q[..., 1::2] - Z.mean(-1, keepdims=True)
                q = q.reshape(Z.shape[0], -1)
            feats.append(q)
            if self.summaries:                                   # quant_tabpfn.py: mean, spread and trend
                feats.append(lean.f_summary(Z).reshape(Z.shape[0], -1))
        return torch.cat(feats, -1)

    def meta(self):
        """Kind, level and shifted flag of every feature, in the column order."""
        out = []
        for a, b, n_pieces, shifted in self.intervals:
            k = n_quantiles(b - a, self.scheme)
            kinds = quantile_kinds(k)
            if self.mean_sub and k > 1:
                kinds = [f"{x} - mean" if j % 2 == 1 else x for j, x in enumerate(kinds)]
            if self.summaries:
                kinds += ["mean", "spread", "trend"]
            out += [{"kind": kind, "level": f"1/{n_pieces}", "shifted": shifted, "position": ""} for kind in kinds]
        return out


QUANT_CFG = dict(depth=6, shifted=True, scheme="quant", mean_sub=True, summaries=False)     # quant.py
V9_CFG = dict(shifted=False, scheme="minmax", mean_sub=False, summaries=True)               # v8; depth: the round's


def quant_features(X, cfg):
    return torch.cat([GenIntervals(r(X).shape[-1], **cfg).transform(r(X)) for r in REPS], -1)


def quant_meta(X, cfg):
    return [dict(m, source=name) for name, r in zip(QUANT_VIEWS, REPS)
            for m in GenIntervals(r(X[:1]).shape[-1], **cfg).meta()]


# == the supervised extractor with other interval features for QUANT's four views ======================================
class TestExtractor(lean.SupervisedExtractor):
    """quant_tabpfn.SupervisedExtractor; qcfg changes the interval features of QUANT's four views (None = v8's)."""

    def __init__(self, qcfg=None, **kw):
        super().__init__(**kw)
        self.qcfg = qcfg

    def make_model(self, name, length, depth):
        if self.qcfg is None or name not in QUANT_VIEWS:
            return GenIntervals(length, depth=depth, **V9_CFG)
        cfg = dict(self.qcfg)
        d = cfg.pop("depth", None)
        return GenIntervals(length, depth=max(depth, d or depth), **cfg)

    def interval_meta(self, name, model, length):
        return [dict(m, source=name) for m in model.meta()]

    def fit_round(self, X, Y, depth):
        """quant_tabpfn.SupervisedExtractor.fit_round, with make_model for the interval features."""
        from aeon.transformations.collection.feature_based import Catch22
        import aeon.transformations.collection.feature_based._catch22 as c22_names
        views = self.views(X)
        self.models = {name: self.make_model(name, Z.shape[-1], depth) for name, Z in views}
        meta = [m for name, Z in views for m in self.interval_meta(name, self.models[name], Z.shape[-1])]
        if self.catch22:
            self.c22, self.c22_k = {}, {}
            for name, Z in (("series", X), ("change (smoothed)", views[1][1])):
                self.c22_k[name] = self.c22_factor(Z)
                self.c22[name] = Catch22(replace_nans=True).fit(self.c22_input(Z, name), Y)
                meta += [{"source": "catch22", "kind": f"{f} ({name})", "level": "1/1", "position": "whole",
                          "shifted": False} for f in c22_names.feature_names]
        if self.templates:
            if "template_set" not in self.round_cache:
                self.fit_templates(X[:, 0, :], Y)
                self.round_cache["template_set"] = True
            meta += [{"source": "template", "kind": "distance", "level": "1/1", "position": "anywhere",
                      "shifted": False} for _ in self.template_owners]
        C = self.candidates(X)
        assert C.shape[1] == len(meta), (C.shape, len(meta))
        self.keep, self.selection_log = self.select(C.numpy(), Y, meta)
        self.n_candidates = C.shape[1]
        self.view_width = {name: sum(m["source"] == name for m in meta) for name, Z in views}
        self.kept_views = {meta[k]["source"] for k in self.keep}
        if self.catch22:
            self.c22_kept, start = {}, len(meta) - 44 - (len(self.template_owners) if self.templates else 0)
            for v, name in enumerate(["series", "change (smoothed)"]):
                columns = [k - start - 22 * v for k in self.keep if start + 22 * v <= k < start + 22 * (v + 1)]
                features = [c22_names.feature_names[c] for c in columns]
                self.c22_kept[name] = (columns, Catch22(features=features, replace_nans=True).fit(
                    self.c22_input(X[:2], "series"), Y[:2]) if features else None)
        return C[:, self.keep]


def extractor(qcfg=None, rank="both"):
    """v9's settings (fill rounds down to 1/16, 250 features, lags auto, catch22, templates), except the quantiles."""
    return TestExtractor(qcfg=qcfg, depth=4, div=8, num_quantiles=2, lags="auto", catch22=True, templates=True,
                         budget=250, fill=True, max_depth=5, c22_shorten=True, rank=rank, seed=0)


def kept_counts(log):
    """What was kept, by kind (the kinds of interval features: QUANT's four views only)."""
    kept = [m for m in log if m["outcome"].startswith("kept")]
    q = [m for m in kept if m["source"] in QUANT_VIEWS]
    return {"kept": len(kept), "kept_quant_views": len(q),
            "kept_shifted": sum(m.get("shifted", False) for m in q),
            "kept_1/16": sum(m["level"] == "1/16" for m in q), "kept_1/32": sum(m["level"] == "1/32" for m in q),
            "kept_median": sum(m["kind"].startswith("median") for m in q),
            "kept_inner": sum(m["kind"].startswith("inner") for m in q),
            "kept_mean_sub": sum(m["kind"].endswith("- mean") for m in q),
            "kept_minmax": sum(m["kind"].split(" ")[0] in ("min", "max") for m in q),
            "kept_summaries": sum(m["kind"] in ("mean", "spread", "trend") for m in q),
            "kept_lag": sum(m["source"].startswith("lag") for m in kept),
            "kept_catch22": sum(m["source"] == "catch22" for m in kept),
            "kept_template": sum(m["source"] == "template" for m in kept)}


VARIANTS = {   # name -> ("quant", interval settings, features for TabPFN; None = all) or ("v9", QUANT views' settings, order)
    "QA QUANT, all features": ("quant", QUANT_CFG, None),
    "Q0 QUANT": ("quant", QUANT_CFG, 250),
    "Q1 QUANT - shifted": ("quant", dict(QUANT_CFG, shifted=False), 250),
    "Q2 QUANT - 1/32": ("quant", dict(QUANT_CFG, depth=5), 250),
    "Q3 QUANT - mean-subtracted": ("quant", dict(QUANT_CFG, mean_sub=False), 250),
    "Q4 QUANT, 20% quantiles (median kept)": ("quant", dict(QUANT_CFG, scheme="q20"), 250),
    "Q5 QUANT, min/max only": ("quant", dict(QUANT_CFG, scheme="minmax"), 250),
    "V0 v9": ("v9", None, "both"),
    "V1 v9 + shifted": ("v9", dict(V9_CFG, shifted=True), "both"),
    "V2 v9 + 1/16 and 1/32 always": ("v9", dict(V9_CFG, depth=6), "both"),
    "V3 v9, 20% quantiles (median kept)": ("v9", dict(V9_CFG, scheme="q20"), "both"),
    "V4 v9, QUANT's intervals + summaries": ("v9", dict(QUANT_CFG, summaries=True), "both"),
    "A0 v9, ANOVA order": ("v9", None, "anova"),
    "A3 v9 + 20% quantiles, ANOVA order": ("v9", dict(V9_CFG, scheme="q20"), "anova"),
    "QB QUANT, 2 x 200": ("quant", QUANT_CFG, 200),
}
GROUPS = {"ablation": [v for v in VARIANTS if v[0] in "QV" and v[1] != "B"],
          "quantiles": ["V0 v9", "V3 v9, 20% quantiles (median kept)"],
          "anova": ["A0 v9, ANOVA order", "A3 v9 + 20% quantiles, ANOVA order"],
          "quant200": ["QB QUANT, 2 x 200"]}
COLUMNS = ["dataset", "resample", "variant", "accuracy", "n_features", "time", "candidates", "kept", "kept_quant_views",
           "kept_shifted", "kept_1/16", "kept_1/32", "kept_median", "kept_inner", "kept_mean_sub", "kept_minmax",
           "kept_summaries", "kept_lag", "kept_catch22", "kept_template", "depth_used"]


# == data, TabPFN, one run =============================================================================================
def load(ts_path, idx_path, dataset, i):
    """The split of run.py: resample 0 = the original split, else the bake-off redux indices."""
    from sklearn.preprocessing import LabelEncoder
    Xtr, Ytr = load_from_tsfile(f"{ts_path}/{dataset}/{dataset}_TRAIN.ts")
    Xte, Yte = load_from_tsfile(f"{ts_path}/{dataset}/{dataset}_TEST.ts")
    if i > 0:
        X, Y = np.concatenate((Xtr, Xte)), np.concatenate((Ytr, Yte))
        tr = np.loadtxt(f"{idx_path}/{dataset}/resample{i}Indices_TRAIN.txt", dtype=np.int32)
        te = np.setdiff1d(np.arange(len(X)), tr)
        Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
    Xtr = torch.tensor(Xtr.squeeze().astype(np.float32)).unsqueeze(1)
    Xte = torch.tensor(Xte.squeeze().astype(np.float32)).unsqueeze(1)
    le = LabelEncoder()
    return Xtr, le.fit_transform(Ytr), Xte, le.transform(Yte)


def tabpfn_accuracy(Ztr, Ytr, Zte, Yte):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(n_estimators=2, device="cuda", random_state=0)
    clf.fit(Ztr, Ytr)
    acc = float((clf.predict_proba(Zte).argmax(1) == Yte).mean())
    del clf
    torch.cuda.empty_cache()
    return acc


def run_variant(name, Xtr, Ytr, Xte, Yte):
    kind, cfg, third = VARIANTS[name]
    t0 = time.perf_counter()
    extra = {}
    if kind == "quant":
        Ztr, Zte = quant_features(Xtr, cfg).numpy(), quant_features(Xte, cfg).numpy()
        extra["candidates"] = Ztr.shape[1]
        if third is not None:                                    # the budget, picked as v4 (ANOVA, then ExtraTrees)
            keep = select_features(Ztr, Ytr, third, 0)
            meta = quant_meta(Xtr, cfg)
            extra.update(kept_counts([dict(meta[k], outcome="kept") for k in keep]))
            Ztr, Zte = Ztr[:, keep], Zte[:, keep]
    else:
        ext = extractor(cfg, third)
        Ztr = ext.fit_transform(Xtr, Ytr).numpy()
        Zte = ext.transform(Xte).numpy()
        extra = dict(kept_counts(ext.selection_log), depth_used=ext.depth_used, candidates=ext.n_candidates)
    acc = tabpfn_accuracy(Ztr, Ytr, Zte, Yte)
    return {"accuracy": acc, "n_features": Ztr.shape[1], "time": time.perf_counter() - t0, **extra}


def check():
    """The interval features of this script = those of quant.py (QUANT) and of quant_tabpfn.py (v8's)."""
    torch.manual_seed(0)
    for length in [24, 84, 150, 500, 637]:
        X = torch.randn(6, 1, length)
        assert torch.equal(orig.Quant().fit_transform(X, np.arange(6) % 2), quant_features(X, QUANT_CFG))
        assert torch.equal(lean.Quant(depth=4, div=8, num_quantiles=2).fit_transform(X, np.arange(6) % 2),
                           quant_features(X, dict(V9_CFG, depth=4)))
        model = lean.IntervalModel(length, depth=4, div=8, fraction=0.2)
        assert torch.equal(torch.cat([model.transform(X)], -1),
                           GenIntervals(length, depth=4, **dict(V9_CFG, scheme="q20")).transform(X))
    print("interval features identical to quant.py and quant_tabpfn.py (incl. --quantile_fraction 0.2): ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-p", "--resample_indices_path")
    parser.add_argument("-d", "--datasets")
    parser.add_argument("-r", "--num_resamples", type=int, default=10)
    parser.add_argument("--variants", choices=list(GROUPS))
    parser.add_argument("-o", "--output_file")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        sys.exit()
    out = Path(args.output_file)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["dataset"], prev["resample"], prev["variant"]))
        print(f"resuming: {len(done)} runs already in {out}")
    datasets = Path(args.datasets).read_text().split()
    for i in range(args.num_resamples):
        for dataset in datasets:
            todo = [v for v in GROUPS[args.variants] if (dataset, i, v) not in done]
            if not todo:
                continue
            data = load(args.input_path, args.resample_indices_path, dataset, i)
            for v in todo:
                row = {"dataset": dataset, "resample": i, "variant": v, **run_variant(v, *data)}
                pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"resample {i} done", flush=True)
