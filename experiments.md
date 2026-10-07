# Experiments

Every results file in `results/` comes from one command below. The versions follow the order of the report:
each one starts from what the previous one found. Names: `{version}[-{variant}]_{classifier settings}_{datasets}_{hardware}.csv`,
e.g. `v3-quant_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv` = v3 on QUANT's features, TabPFN with 2 passes
(`e2`) and 200 features (`f200`), the model cache, the small datasets, all CPU cores + the GPU. Without classifier
settings in the name, the file has one row per classifier (`classifier` column).

| Version | What it is | Notebook |
|---|---|---|
| **v0** | the authors' code, unmodified | 01 |
| **v1** | QUANT with timers, metrics, a seed and a choice of classifier (ExtraTrees, XGBoost, CatBoost, TabPFN) | 01, 02, 03, 06 |
| **v2** | QUANT + extra features (lag-d differences, MultiRocket, RDST, MR-Hydra), filtered on the training data | 03 |
| **v3** | TabPFN's cost settings (passes, features, training cases, model cache) on QUANT's and v2's features | 04, 05 |
| **v4** | the lean extractor for TabPFN (`quant_tabpfn.py`) and how TabPFN's features are picked: the "cheap TabPFN" | 04, 05, 06 |
| **v5** | the supervised extractor: it picks at most 250 features itself (selection, shape view, catch22, templates) | 05 |
| **v6** | v5 refined: lag-16 only, the budget filled, catch22 on a shortened copy | 05 |
| **v7** | v6 with ExtraTrees importance in the selection and a smaller budget for large datasets | 05 |
| **v8** | v7 with the lag-d differences chosen per dataset | 05 |
| **v9** | v8 with two speed fixes, 200 features on every dataset (no size rule) and a fifth of QUANT's quantiles (median kept): the final version | 05, 06 |

## How to run them

- In Windows `cmd`, from the folder that contains `quant-reproduction`.
- `.venv` (CPU PyTorch) for the reproduction runs (v0, v1 on one core); `.venv-gpu` (CUDA PyTorch, XGBoost, CatBoost, TabPFN,
  aeon) for everything else.
- One-core runs: run the `set` lines below before the commands (the other runs in a new window).

| Option | Meaning |
|---|---|
| `-d` | the dataset list: `original\code\ucr142.txt` (142 datasets), or in `improvedQuant\dataSetLists`: `smoke-small.txt`, `smoke-medium.txt` (5 datasets each), `worst3.txt`, `worst7.txt`, `worst10.txt` (the datasets where a version loses most to QUANT + ExtraTrees), `deep6.txt` (the datasets whose fill rounds reached 1/32 in v8), `chlorine-uwave.txt` |
| `-r` | the number of resamples: `1` = resample 0 only, `3` = resamples 0 to 2, `30` = all |
| `-c` | the classifiers: `extratrees`, `xgboost`, `catboost`, `tabpfn`, or `all` |
| `-j 1` | one CPU core (default: all cores) |
| `--device cpu` | XGBoost, CatBoost and TabPFN on the CPU (default: the GPU) |
| `-s` | the seed (default 0) |
| `-o` | the results file |

XGBoost and CatBoost skip runs with more than 14,000 features (`--xgboost_max_features`, `--catboost_max_features` controls feature size),
and the skip is recorded in the results file: on a 6 GB GPU, larger feature sets ran out of memory or hung the computer.

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

## v1: QUANT with timers and a choice of classifier (notebooks 01, 02, 03, 06)

`improvedQuant\v1_classifiers\run.py`: the authors' method (quant.py unchanged) with per-phase timers, more
metrics, seed 0 and a choice of classifier. On the published setup, one CPU core (notebook 01; after the one-core
`set` lines):

```bat
quant-reproduction\.venv\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -j 1 -c extratrees -s 0 -o quant-reproduction\results\v1_ucr142_onecore-cpu.csv
```

The same on all CPU cores, for the time comparison of notebook 06 (identical accuracies; a new window):

```bat
quant-reproduction\.venv\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c extratrees -s 0 --device cpu -o quant-reproduction\results\v1_ucr142_allcores-cpu.csv
```

The four classifiers on one core and on all cores, with and without the GPU, resample 0 (notebook 02). One core +
GPU and one core CPU only (after the one-core `set` lines; CPU only on the small datasets, TabPFN would take hours
on the medium ones), then all cores + GPU (a new window):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 1 -c all -j 1 -o quant-reproduction\results\v1_smoke-small-r0_onecore-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 1 -c all -j 1 -o quant-reproduction\results\v1_smoke-medium-r0_onecore-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 1 -c all -j 1 --device cpu -o quant-reproduction\results\v1_smoke-small-r0_onecore-cpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 1 -c all -o quant-reproduction\results\v1_smoke-small-r0_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 1 -c all -o quant-reproduction\results\v1_smoke-medium-r0_allcores-gpu.csv
```

The four classifiers, 3 resamples, all cores + GPU: the reference for v2 and v3 (notebooks 03, 04):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all -o quant-reproduction\results\v1_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\v1_classifiers\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all -o quant-reproduction\results\v1_smoke-small_allcores-gpu.csv
```

## v2: QUANT + extra features (notebook 03)

`improvedQuant\features\run.py` (the options are explained at the top of the file). The extra features are
filtered on the training data (ANOVA F-score). `lagd`: lag-d differences x(t) - x(t-d), d = 2, 4, 8, 16, summarised
with QUANT's interval quantiles, all QUANT features kept + the best lag-d ones; `lagd-joint`: QUANT and lag-d
filtered together back to QUANT's count; `multirocket`, `rdst`: lag-d + the best MultiRocket features or RDST
shapelets; `mrhydra`: the best MultiRocket + Hydra features. All four classifiers, 3 resamples:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter separate -o quant-reproduction\results\v2-lagd_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter separate -o quant-reproduction\results\v2-lagd_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter joint -o quant-reproduction\results\v2-lagd-joint_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter joint -o quant-reproduction\results\v2-lagd-joint_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --lags 2,4,8,16 --extra multirocket --lag_filter separate --extra_test selected -o quant-reproduction\results\v2-multirocket_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra multirocket --lag_filter separate --extra_test selected -o quant-reproduction\results\v2-multirocket_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --lags 2,4,8,16 --extra rdst --lag_filter separate --extra_test selected -o quant-reproduction\results\v2-rdst_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra rdst --lag_filter separate --extra_test selected -o quant-reproduction\results\v2-rdst_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --extra mrhydra --prune filter -o quant-reproduction\results\v2-mrhydra_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --extra mrhydra --prune filter -o quant-reproduction\results\v2-mrhydra_smoke-small_allcores-gpu.csv
```

## v3: TabPFN's cost settings (notebooks 04, 05)

TabPFN only, 3 resamples. `e` = passes (`--tabpfn_estimators`), `f` = features (`--tabpfn_features`, the most
important ones in an ExtraTrees fit on the training data), `c1000` = at most 1000 training cases (`--tabpfn_context`,
medium datasets only), `cache` = TabPFN's model built once (`--tabpfn_cache`). On QUANT's features (`v3-quant`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e1f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e1f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v3-quant_tabpfn-e2f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v3-quant_tabpfn-e2f200_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 500 --tabpfn_context 1000 -o quant-reproduction\results\v3-quant_tabpfn-e2f500-c1000_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e2f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e2f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 4 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e4f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 4 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e4f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 200 -o quant-reproduction\results\v3-quant_tabpfn-e8f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 200 -o quant-reproduction\results\v3-quant_tabpfn-e8f200_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_context 1000 -o quant-reproduction\results\v3-quant_tabpfn-e8f500-c1000_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e8f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 -o quant-reproduction\results\v3-quant_tabpfn-e8f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e1f500-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 1 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e1f500-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e2f200-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e8f500-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e8f500-cache_smoke-small_allcores-gpu.csv
```

On v2-lagd's features (`v3-lagd`; the `e2f200` files also hold the other three classifiers, `-c all`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 100 -o quant-reproduction\results\v3-lagd_tabpfn-e1f100_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 100 -o quant-reproduction\results\v3-lagd_tabpfn-e1f100_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 200 -o quant-reproduction\results\v3-lagd_tabpfn-e1f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 200 -o quant-reproduction\results\v3-lagd_tabpfn-e1f200_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v3-lagd_tabpfn-e1f300-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v3-lagd_tabpfn-e1f300-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 300 -o quant-reproduction\results\v3-lagd_tabpfn-e1f300_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 300 -o quant-reproduction\results\v3-lagd_tabpfn-e1f300_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 400 -o quant-reproduction\results\v3-lagd_tabpfn-e1f400_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 400 -o quant-reproduction\results\v3-lagd_tabpfn-e1f400_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v3-lagd_tabpfn-e1f500_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 1 --tabpfn_features 500 -o quant-reproduction\results\v3-lagd_tabpfn-e1f500_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v3-lagd_tabpfn-e2f200-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v3-lagd_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v3-lagd_tabpfn-e2f200_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c all --lags 2,4,8,16 --extra none --lag_filter separate --tabpfn_estimators 2 --tabpfn_features 200 -o quant-reproduction\results\v3-lagd_tabpfn-e2f200_smoke-small_allcores-gpu.csv
```

Diagnostics: TabPFN on QUANT's features for the datasets where a later version loses most to QUANT + ExtraTrees,
30 resamples. The 3 worst datasets of v4 (notebook 05) with 2 x 200 and 8 x 500, and the 10 worst of v8 (notebook 05,
section 11) with all of QUANT's features (`fall`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e2f200-cache_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --tabpfn_estimators 8 --tabpfn_features 500 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e8f500-cache_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst10.txt -r 30 -c tabpfn --tabpfn_estimators 2 --tabpfn_features 20000 --tabpfn_cache -o quant-reproduction\results\v3-quant_tabpfn-e2fall-cache_worst10_allcores-gpu.csv
```

## v4: a lean extractor for TabPFN, the "cheap TabPFN" (notebooks 04, 05, 06)

`--quant tabpfn` = `improvedQuant\features\quant_tabpfn.py`: fewer and less redundant features (its header lists
the changes to quant.py). `v4-first`: the first version (depth 5, one quantile per 8 points) on v2-lagd's settings.
`v4`: 4 levels of pieces, 2 quantiles per piece (min and max) + mean, spread and trend, lag-d 4 and 16, about 450
features; TabPFN's features picked by ExtraTrees importance, ANOVA, or both (ANOVA drops the weaker part, then
ExtraTrees picks); `--feature_log` also writes `<results file>_features.csv`. `v4-nofilter`: about 252 features
(3 levels, 3 quantiles), all given to TabPFN. `v4-wide`: about 1,800 features, ANOVA alone picks 200. 3 resamples:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --quant tabpfn --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-first_tabpfn-e1f300-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --quant tabpfn --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-first_tabpfn-e1f300-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --quant tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-first_tabpfn-e2f200-cache_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --lag_filter separate --quant tabpfn --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-first_tabpfn-e2f200-cache_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank anova --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-anova_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank anova --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-anova_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank both --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-both_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank both --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-both_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank extratrees --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-extratrees_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank extratrees --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e1f300-extratrees_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank anova --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-anova_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank anova --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-anova_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-both_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-both_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank extratrees --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-extratrees_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank extratrees --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-extratrees_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --prune none --quant tabpfn -m 3 --quantiles 3 --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-nofilter_tabpfn-e1_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --prune none --quant tabpfn -m 3 --quantiles 3 --tabpfn_estimators 1 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-nofilter_tabpfn-e1_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 4,16 --extra none --prune none --quant tabpfn -m 3 --quantiles 3 --tabpfn_estimators 2 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-nofilter_tabpfn-e2_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 4,16 --extra none --prune none --quant tabpfn -m 3 --quantiles 3 --tabpfn_estimators 2 --tabpfn_features 300 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-nofilter_tabpfn-e2_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --prune none --quant tabpfn --tabpfn_rank anova --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-wide_tabpfn-e2f200-anova_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --lags 2,4,8,16 --extra none --prune none --quant tabpfn --tabpfn_rank anova --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4-wide_tabpfn-e2f200-anova_smoke-small_allcores-gpu.csv
```

The chosen setting (2 passes x 200 features, both, model cache) on the published setup, 142 datasets x 30
resamples, all cores + GPU (about 3.5 hours; notebook 06). Every run is written as soon as it finishes: to stop,
press Ctrl+C; to continue, run the same command again (the runs already in the file are skipped):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn --lags 4,16 --extra none --lag_filter separate --quant tabpfn -m 4 --quantiles 2 --tabpfn_rank both --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v4_tabpfn-e2f200-both_ucr142_allcores-gpu.csv
```

Diagnostic: ExtraTrees on v4's features (all 450) for the 3 datasets where v4 loses most, 30 resamples (notebook 05):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c extratrees --quant tabpfn -m 4 --quantiles 2 --lags 4,16 -o quant-reproduction\results\v4_extratrees_worst3_allcores-cpu.csv
```

## v5: a supervised extractor (notebook 05)

`--quant supervised` = `quant_tabpfn.SupervisedExtractor`: a pool of candidate features (the interval features of
QUANT's four views and lag-d 4 and 16), a relevance check (ANOVA, p < 0.05) and a greedy selection of at most 250
features without near-copies; TabPFN gets all of them. Each step adds to the previous one: `sel` the selection
alone, `shape` + a z-normalized view (`--znorm`), `catch22` + catch22 (`--catch22`), `templates` + pattern
templates (`--templates`). `--feature_log` writes `<results file>_selection.csv`: what happened to every
candidate. The 3 worst datasets of v4 (30 resamples) and both smoke lists (3 resamples):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-sel_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --quant supervised --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-sel_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-sel_tabpfn-e2f250_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --znorm --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-shape_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --quant supervised --znorm --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-shape_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --znorm --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-shape_tabpfn-e2f250_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --znorm --catch22 --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-catch22_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --quant supervised --znorm --catch22 --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-catch22_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --znorm --catch22 --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-catch22_tabpfn-e2f250_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --znorm --catch22 --templates --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-templates_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 3 -c tabpfn --quant supervised --znorm --catch22 --templates --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-templates_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --znorm --catch22 --templates --lags 4,16 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v5-templates_tabpfn-e2f250_worst3_allcores-gpu.csv
```

## v6: the supervised extractor refined (notebook 05)

v5-templates without the z-normalized view and lag-4 (both mostly repeat another view), `--fill_budget` (the
selection goes on past p >= 0.05 until 250 features are kept; if even that is not enough, finer pieces), and
`--c22_shorten` (catch22 on a shortened copy of long, smooth series). The small list with 30 resamples:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v6_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v6_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v6_tabpfn-e2f250_worst3_allcores-gpu.csv
```

## v7: ExtraTrees importance in the selection (notebook 05)

v6 with `--supervised_rank both` (ANOVA keeps the budget + half of the rest, then the ExtraTrees importance
decides the order of the greedy selection, as v4's "both"), and a smaller budget for large datasets
(`--relative_budget --relative_min 150`: above 2000 training + test series the budget shrinks to 250 x 2000 /
series, at least 150):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v7_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v7_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v7_tabpfn-e2f250_worst3_allcores-gpu.csv
```

## v8: the lag-d differences chosen per dataset (notebook 05)

v7 with `--lags auto`: lag-d views for d = 2, 4, 8, ... up to a quarter of the series length (at most 64) all go
into the candidate pool, the selection decides which are used; new data computes only the views with kept
features. `v8-lag4`: v7 with lag-4 added back instead (`--lags 4,16`):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --catch22 --templates --lags 4,16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8-lag4_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 4,16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8-lag4_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags 4,16 --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8-lag4_tabpfn-e2f250_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 3 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8_tabpfn-e2f250_worst3_allcores-gpu.csv
```

On the published setup, 142 datasets x 30 resamples (about 5 hours; notebook 05, section 11; it can be stopped and continued
as v4's). Its selection log (`_selection.csv`, about 200 MB) is too large for GitHub and is not in the repository:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8_tabpfn-e2f250_ucr142_allcores-gpu.csv
```

Diagnostics for the 10 datasets where v8 loses most to QUANT + ExtraTrees, 30 resamples (notebook 05, section 11): v8's
features with TabPFN and with ExtraTrees, and v8 without `--fill_budget`:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst10.txt -r 30 -c tabpfn extratrees --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8_tabpfn-e2f250-and-extratrees_worst10_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst10.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache --feature_log -o quant-reproduction\results\v8-nofill_tabpfn-e2f250_worst10_allcores-gpu.csv
```

## v9: the final version (notebooks 05, 06)

v8 with two speed fixes, no size rule and more quantiles. (1) catch22 and the templates do not depend on the piece
size, so they are computed once and shared by the fill rounds (always on in the code; identical results, about 2x
faster on long series). (2) `--supervised_max_depth 5`: the fill rounds stop at pieces of 1/16 of the series (v8
also tried 1/32, whose features were rarely kept). No `--relative_budget`: every dataset gets 200 features
(`--supervised_budget 200`; on the small and medium datasets 200 were as accurate as 250, and faster).
(3) `--quantile_fraction 0.2`: QUANT's four views get a fifth of QUANT's quantiles (one per 4 points), odd so that
the median stays in, instead of only the minimum and maximum (the lag-d views keep min and max). Of the features
QUANT has and v8 does not (the 1/32 pieces, shifted pieces, the median and inner quantiles, quantiles minus the
piece's mean), only the median and inner quantiles helped TabPFN in a test on the 7 datasets where v8 loses most to
QUANT + ExtraTrees (10 resamples): +0.009 there, +0.005 on the 3 worst datasets of v4, no change on the small and
medium datasets on average (UWaveGestureLibraryX +0.019, ChlorineConcentration -0.018); notebook 05, section 12.
142 datasets x 30
resamples, about 4.5 to 5 hours (on the largest datasets 200 features make TabPFN slower than v8's 150):

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\original\code\ucr142.txt -r 30 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --supervised_rank both --supervised_max_depth 5 --quantile_fraction 0.2 --supervised_budget 200 --tabpfn_estimators 2 --tabpfn_features 200 --tabpfn_cache --feature_log -o quant-reproduction\results\v9_tabpfn-e2f200_ucr142_allcores-gpu.csv
```

The tests behind v9 (notebook 05, section 12). The depth of the fill rounds: v8 with the fill rounds stopped at 1/16
(`--supervised_max_depth 5`) on the 15 datasets whose fill rounds reached 1/32 in v8's full run (`deep6.txt`), 10
resamples:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\run.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\deep6.txt -r 10 -c tabpfn --quant supervised --catch22 --templates --lags auto --fill_budget --c22_shorten --relative_budget --relative_min 150 --supervised_rank both --supervised_max_depth 5 --tabpfn_estimators 2 --tabpfn_features 250 --tabpfn_cache -o quant-reproduction\results\v8-depth5_tabpfn-e2f250_deep6_allcores-gpu.csv
```

Which of QUANT's features help TabPFN: `improvedQuanteatureseature_test.py` switches QUANT's kinds of interval
features on and off one at a time, one row per run and variant (`variant` column). `ablation`: on the 7 datasets where
v8 loses significantly to QUANT + ExtraTrees (`worst7.txt`), QUANT + TabPFN (2 passes x 250 features picked as v4, or
all) with one kind removed or changed, and the supervised extractor with v9's settings with one kind added;
`quantiles`: v9's settings with min and max or a fifth of QUANT's quantiles; `anova`: the same with the ANOVA order;
`quant200`: QUANT + TabPFN with 2 passes x 200 features picked as v4 (the reference for v9's 200 features).
`feature_test.py --check` checks that its features equal those of quant.py and quant_tabpfn.py:

```bat
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst7.txt -r 10 --variants ablation -o quant-reproduction\results\v9-ablation_tabpfn_worst7_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\worst3.txt -r 30 --variants quantiles -o quant-reproduction\results\v9-quantiles_tabpfn-e2f250_worst3_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 --variants quantiles -o quant-reproduction\results\v9-quantiles_tabpfn-e2f250_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 10 --variants quantiles -o quant-reproduction\results\v9-quantiles_tabpfn-e2f250_smoke-medium_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\chlorine-uwave.txt -r 10 --variants anova -o quant-reproduction\results\v9-anova_tabpfn-e2f250_chlorine-uwave_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-small.txt -r 30 --variants quant200 -o quant-reproduction\results\v3-quant_tabpfn-e2f200-both_smoke-small_allcores-gpu.csv
quant-reproduction\.venv-gpu\Scripts\python.exe quant-reproduction\improvedQuant\features\feature_test.py -i quant-reproduction\data\ts -p quant-reproduction\data\indices -d quant-reproduction\improvedQuant\dataSetLists\smoke-medium.txt -r 10 --variants quant200 -o quant-reproduction\results\v3-quant_tabpfn-e2f200-both_smoke-medium_allcores-gpu.csv
```
