# Experiments

Every results file in `results/` comes from one command below. The versions follow the order of the report: each one
starts from what the previous one found. Names: `{version}[-{variant}]_{classifier settings}_{datasets}_{hardware}.csv`,
e.g. `v2_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv` = v2 on QUANT's features, TabPFN with 2 passes
(`e2`) and 200 features (`f200`), the model cache, the small datasets, all CPU cores + the GPU. Without classifier
settings in the name, the file has one row per classifier (`classifier` column).

| Version | What it is | Notebook |
|---|---|---|
| **v0** | the authors' code, unmodified | 01 |
| **v1** | QUANT with timers, metrics, a seed and a choice of classifier (ExtraTrees, XGBoost, CatBoost, TabPFN) | 01, 02, 03, 04, 05 |
| **v2** | TabPFN's cost settings (passes, features, the model cache) on QUANT's features | 03, 04 |
| **v3** | the lean extractor for TabPFN (`quant_tabpfn.py`): the "cheap TabPFN" | 03, 04, 05 |
| **v4** | the supervised extractor (`quant_tabpfn.SupervisedExtractor`): the final version, and its ablation | 04, 05 |

Earlier development steps are not kept: other feature families next to QUANT's (MultiRocket, RDST, MR-Hydra, lag-d
differences; only the lag-d differences helped, and v3 and v4 keep them), and the step-by-step build of the
supervised extractor; the ablation of v4 (notebook 04, section 3) shows what each of its parts adds.

## How to run them

- In Windows `cmd`, from the folder that contains `quant-reproduction`.
- `.venv` (CPU PyTorch) for the reproduction runs (v0, v1 on one core); `.venv-gpu` (CUDA PyTorch, XGBoost, CatBoost,
  TabPFN, aeon) for everything else.
- One-core runs: run the `set` lines below before the commands (the other runs in a new window).
- Every run is written to the results file as soon as it finishes: to stop, press Ctrl+C; to continue, run the same
  command again (the runs already in the file are skipped).

| Option | Meaning |
|---|---|
| `-d` | the dataset list: `original\code\ucr142.txt` (142 datasets), or in `improvedQuant\dataSetLists`: `smoke-small.txt`, `smoke-medium.txt` (5 datasets each), `worst3.txt` (the 3 datasets where v3 loses most to QUANT + ExtraTrees), `dev34.txt` (the 34 datasets used to develop v3 and v4), `heldout108.txt` (the other 108) |
| `-r` | the number of resamples: `1` = resample 0 only, `3` = resamples 0 to 2, `10` = 0 to 9, `30` = all |
| `-c` | the classifiers: `extratrees`, `xgboost`, `catboost`, `tabpfn`, or `all` (`v1_classifiers\run.py`); `extratrees`, `tabpfn` (`features\run.py`) |
| `-j 1` | one CPU core (default: all cores) |
| `--device cpu` | XGBoost, CatBoost and TabPFN on the CPU (default: the GPU) |
| `-s` | the seed (default 0) |
| `-o` | the results file |

XGBoost and CatBoost (v1) skip runs with more than 14,000 features (`--xgboost_max_features`,
`--catboost_max_features`), and the skip is recorded in the results file: on a 6 GB GPU, larger feature sets ran out
of memory or hung the computer.

**One CPU core**, as in the paper's timing claim

```bat
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set LOKY_MAX_CPU_COUNT=1
```

## v0: the authors' code (notebook 01)

142 datasets x 30 resamples on one CPU core (about 6 hours): first the one-core `set` lines, then in the same
window. The authors' `run.py` always writes `results.csv`, so the file is renamed afterwards:

```bat
quant-reproduction\.venv\Scripts\python.exe quant-reproduction\original\code\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -o quant-reproduction\results
ren quant-reproduction\results\results.csv v0_ucr142_onecore-cpu.csv
```

## v1: QUANT with timers and a choice of classifier (notebooks 01, 02, 03, 04, 05)

`improvedQuant\v1_classifiers\run.py`: the authors' method (quant.py unchanged) with per-phase timers, more
metrics, seed 0 and a choice of classifier.

The published setup, the same for v0 and v1 (notebook 01): the 142 datasets of the paper x 30 resamples (0 = the
original UCR split, 1-29 = the bake-off redux splits in `data\indices`); QUANT's defaults (depth 6, div 4) and
ExtraTrees' (200 trees, `max_features` 0.1, entropy); one CPU core, with PyTorch, numpy and ExtraTrees each limited
to one thread; seed: none in v0 (the authors' `run.py` fixes none, so v0 cannot match the published runs exactly),
0 in v1. On one CPU core (after the one-core `set` lines):

```bat
quant-reproduction\.venv\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -j 1 -c extratrees -s 0 -o quant-reproduction\results\v1_ucr142_onecore-cpu.csv
```

The same on all CPU cores, for the time comparison of notebook 05 (identical accuracies; a new window):

```bat
quant-reproduction\.venv\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c extratrees -s 0 --device cpu -o quant-reproduction\results\v1_ucr142_allcores-cpu.csv
```

The four classifiers, all cores + GPU, resample 0 (notebook 02) and resamples 0 to 2 (the reference of notebook 03):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 1 -c all -o quant-reproduction\results\v1_smoke-small-r0_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 1 -c all -o quant-reproduction\results\v1_smoke-medium-r0_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all -o quant-reproduction\results\v1_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all -o quant-reproduction\results\v1_smoke-small_allcores-gpu.csv
```

## v2: TabPFN's cost settings (notebooks 03, 04)

`improvedQuant\features\run.py` on QUANT's own features (`--quant original`, the default), TabPFN only, 3 resamples.
`e` = passes (`--tabpfn_estimators`), `f` = features (`--tabpfn_features`, the most important ones in an ExtraTrees
fit on the training data), `cache` = TabPFN's model built once (`--tabpfn_cache`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e1f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e1f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v2_tabpfn-e2f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v2_tabpfn-e2f200_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e2f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e2f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 4 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e4f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 4 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e4f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 200 -o quant-reproduction\results\v2_tabpfn-e8f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 200 -o quant-reproduction\results\v2_tabpfn-e8f200_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e8f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 -o quant-reproduction\results\v2_tabpfn-e8f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e1f500-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e1f500-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e2f200-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e8f500-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e8f500-cache_smoke-small_allcores-gpu.csv
```

Diagnostic: TabPFN on QUANT's features for the 3 datasets where v3 loses most to QUANT + ExtraTrees, 30 resamples,
2 x 200 and 8 x 500 (notebook 04, section 2):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e2f200-cache_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v2_tabpfn-e8f500-cache_worst3_allcores-gpu.csv
```

## v3: a lean extractor for TabPFN, the "cheap TabPFN" (notebooks 03, 04, 05)

`--quant tabpfn` = `improvedQuant\features\quant_tabpfn.py` (its header lists the changes): 4 levels of pieces, 2
quantiles per piece (min and max) + mean, spread and trend, QUANT's four views and the lag-4 and lag-16 differences,
about 450 features; ANOVA drops the weaker part, then the ExtraTrees importance picks the 200 TabPFN gets
(`--tabpfn_rank both`); 2 passes, the model cache. The small and medium datasets, 3 resamples (notebook 03):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant tabpfn -m 4 --quantiles 2 --lags 4,16 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3_tabpfn-e2f200-both_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --quant tabpfn -m 4 --quantiles 2 --lags 4,16 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3_tabpfn-e2f200-both_smoke-small_allcores-gpu.csv
```

On the published setup, 142 datasets x 30 resamples (about 3 hours; notebook 05):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn --quant tabpfn -m 4 --quantiles 2 --lags 4,16 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3_tabpfn-e2f200-both_ucr142_allcores-gpu.csv
```

Diagnostic: ExtraTrees on v3's features (all 450) for the 3 datasets where v3 loses most, 30 resamples (notebook 04,
section 2):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c extratrees --quant tabpfn -m 4 --quantiles 2 --lags 4,16 -o quant-reproduction\results\v3_extratrees_worst3_allcores-cpu.csv
```

## v4: the supervised extractor, the final version (notebooks 04, 05)

`--quant supervised` = `quant_tabpfn.SupervisedExtractor`: candidates from QUANT's four views (with a fifth of QUANT's
quantiles, median kept), lag-d views (d = 2, 4, 8, ... chosen per dataset), catch22 and pattern templates; a relevance
check (ANOVA) and a greedy selection of at most 200 features without near-copies, ordered by the ExtraTrees
importance; the budget filled with finer pieces down to 1/16 if needed; TabPFN (2 passes, model cache) gets all of
them. On the published setup, 142 datasets x 30 resamples (about 4.7 hours). `--feature_log` also writes
`_selection.csv`, what happened to every candidate; it is about 200 MB, too large for GitHub, so the repository has
the summary that notebook 04 makes from it (`_selection-summary.csv`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200_ucr142_allcores-gpu.csv
```

The ablation (notebook 04, section 3): v4 with one part switched off at a time, on the 34 development datasets,
resamples 0 to 9 (about 2 hours for all six):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-catch22_tabpfn-e2f200_dev34_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --catch22 --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-templates_tabpfn-e2f200_dev34_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --catch22 --templates --lags none --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-lags_tabpfn-e2f200_dev34_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --catch22 --templates --lags auto --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-fill_tabpfn-e2f200_dev34_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank anova --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-rank_tabpfn-e2f200_dev34_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\dev34.txt -r 10 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v4-abl-quantiles_tabpfn-e2f200_dev34_allcores-gpu.csv
```

The default split again with five seeds, for the seed check, the extra metrics (macro F1, Brier score), the peak
memory and the confusion matrices (notebook 05, sections 4b, 7b, 8b, 8c): v4 and QUANT + ExtraTrees on resample 0,
seeds 0 to 4, all in one results file per method (the `seed` column); `--save_predictions` also writes every run's test
labels and class probabilities (`_predictions\`, 1,420 small files; only the six that notebook 05, section 8c, draws are in the repository (resample 0, seed 0 of
ChlorineConcentration, Ham and ECG5000): running the commands writes them all again; only the confusion matrices of notebook 05, section 8c, need them). About 50 minutes for v4 and 20 for QUANT + ExtraTrees. Seed 0
repeats resample 0 of the full runs exactly (notebook 05, section 7b):

```bat
for %s in (0 1 2 3 4) do quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 1 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -s %s --save_predictions -o quant-reproduction\results\v4-seeds_tabpfn-e2f200_ucr142-r0_allcores-gpu.csv
for %s in (0 1 2 3 4) do quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 1 -c extratrees -s %s --save_predictions -o quant-reproduction\results\v1-seeds_extratrees_ucr142-r0_allcores-cpu.csv
```

(In a `.bat` file, write `%%s` instead of `%s`.)

v4 on one CPU core without the GPU (`-j 1 --device cpu`), the default split only, for a time comparable with the
published single-core times (notebook 05, section 6d):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 1 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -j 1 --device cpu -o quant-reproduction\results\v4_tabpfn-e2f200_ucr142-r0_onecore-cpu.csv
```

## The simple baselines (notebook 05, sections 2b, 6 and 8b)

`improvedQuant\baselines\run.py`: the majority class, ExtraTrees on the raw values and catch22 + a random forest
(aeon's Catch22Classifier), on the same 142 datasets x 30 resamples (about an hour, all CPU cores):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\baselines\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -o quant-reproduction\results\baselines_ucr142_allcores-cpu.csv
```

TabPFN on the raw values, the same classifier as v4 (2 passes, the model built once, the GPU) without v4's extractor;
TabPFN subsamples the time points of long series itself (all cores + GPU; 13.6 hours of fit + predict). The laptop
crashed once during the run (a BIOS problem, fixed); the run resumed from the file. The six EthanolLevel runs just before
the crash (resamples 0-5) were about 9 times slower than normal, so they were deleted and run again (same accuracies):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\baselines\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn -o quant-reproduction\results\baselines-tabpfn_ucr142_allcores-gpu.csv
```

## The published results of the bake-off redux (notebook 05, section 6)

Not a run of ours: the accuracies the bake-off redux published for 11 classifiers (112 UCR datasets x 30 resamples),
and their fit and predict times (milliseconds, one CPU core per run on their cluster; notebook 05, section 6d), one
file per classifier and measure from timeseriesclassification.com, saved unchanged in `results\bakeoff-redux\` (`MR` =
MultiROCKET). Downloaded in `cmd` with curl (part of Windows 10 and 11), or one by one in a browser from the same
addresses:

```bat
mkdir quant-reproduction\results\bakeoff-redux
for %n in (HC2 MR-Hydra MR MiniROCKET ROCKET RDST InceptionTime DrCIF FreshPRINCE QUANT 1NN-DTW) do curl -f -o quant-reproduction\results\bakeoff-redux\%n_accuracy.csv https://timeseriesclassification.com/results/ReferenceResults/classification/accuracy/%n_accuracy.csv
for %n in (HC2 MR-Hydra MR MiniROCKET ROCKET RDST InceptionTime DrCIF FreshPRINCE QUANT 1NN-DTW) do curl -f -o quant-reproduction\results\bakeoff-redux\%n_fittime.csv https://timeseriesclassification.com/results/ReferenceResults/classification/fittime/%n_fittime.csv
for %n in (HC2 MR-Hydra MR MiniROCKET ROCKET RDST InceptionTime DrCIF FreshPRINCE QUANT 1NN-DTW) do curl -f -o quant-reproduction\results\bakeoff-redux\%n_predicttime.csv https://timeseriesclassification.com/results/ReferenceResults/classification/predicttime/%n_predicttime.csv
```
