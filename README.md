# Reproducing and Improving QUANT

Reproduction study of

> Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). **QUANT: A Minimalist Interval Method for Time Series Classification.** *Data Mining and Knowledge Discovery.* [doi:10.1007/s10618-024-01036-9](https://doi.org/10.1007/s10618-024-01036-9) · [arXiv:2308.00928](https://arxiv.org/abs/2308.00928) · [original code](https://github.com/angus924/quant)

COMP41850 AI for Time Series, group project. Report: *TODO link*.

## Repository structure

```
original/                       authors' code and published results, unmodified (GPL-3.0)
improvedQuant/                  our modified copy of QUANT
notebooks/                      analysis: our results vs the paper, original vs improved
results/                        results of our runs, one folder per run
data/                           datasets and resample indices (not in git, see section 2)
requirements-original-lock.txt  exact package versions used for the reproduction runs
requirements.txt                packages needed for the improved version
```

## 1. Environment

Python 3.12, CPU build of PyTorch.

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

To reproduce the exact environment of our reproduction runs:

```powershell
uv pip install -r requirements-original-lock.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## 2. Data

Links from the [bake-off notebook](https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/publications/y2023/tsc_bakeoff/tsc_bakeoff_2023.ipynb) referenced in `original/doc.md`.

**Datasets** -> `data/ts/<Name>/<Name>_TRAIN.ts`, `<Name>_TEST.ts`

- 112 UCR datasets ("Univariate aeon formatted ts files"): https://www.timeseriesclassification.com/dataset.php
- 30 new bake-off datasets: https://drive.google.com/file/d/1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3/view

**Resample indices** -> `data/indices/<Name>/resample{i}Indices_TRAIN.txt`

- 112 UCR datasets (Java indices): https://drive.google.com/file/d/1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3/view
- 30 new bake-off datasets (Python indices): https://drive.google.com/file/d/1aLBP_nhnoqz075puKg30zuF3F_QBOXYM/view

Keep only the 142 names in `original/code/ucr142.txt`, and rename `_eq` / `_nmv` / `_disc` to `Eq` / `Nmv` / `Discrete`.

## 3. Reproduce the results

All commands are run from this folder (`quant-reproduction`).

| # | Experiment | Command | Output |
|---|---|---|---|
| 0 | Smoke test: original code, 5 datasets x 3 resamples | `.venv\Scripts\python.exe original\code\run.py -i data/ts -p data/indices -d improvedQuant/dataSetLists/smoke5.txt -r 3 -o results/smoke_original` | `results/smoke_original/results.csv` |
| 1 | Main result: original code, 142 datasets x 30 resamples | `.venv\Scripts\python.exe original\code\run.py -i data/ts -p data/indices -d original/code/ucr142.txt -r 30 -o results/original_full_run1` | `results/original_full_run1/results.csv` |
| 2 | Comparison with the paper | open `notebooks/01_compare_with_paper.ipynb`, set `OUR_RESULTS`, run all cells | tables and plots |

Notes:
- Create the `-o` folder first (`mkdir results\smoke_original`). `run.py` writes `results.csv` there and overwrites an existing one.
- `run.py` saves only at the end of the run (experiment 1 takes a few hours).
- `run.py` does not seed ExtraTrees, so accuracies differ slightly from run to run and from the published values.
- A dataset list with only one name makes `run.py` crash; use at least two.

## 4. Results

*TBD: our numbers vs the paper's reported numbers.*

| | Paper | Bake-off redux (QUANT) | Ours |
|---|---|---|---|
| Mean accuracy, 142 datasets x 30 resamples | 0.8568 | TBD | TBD |
| Mean accuracy, 112 original datasets | 0.8671 | TBD | TBD |
| Total compute time per resample, 1 core | < 15 min (14.86 in the published CSV) | - | TBD |

## 5. Improvements

*TBD* - built in `improvedQuant/`:
- per-phase timers and single-core timing
- QUANT for unequal-length series
- when and why QUANT works or fails
- GPU classifiers (XGBoost, CatBoost) instead of ExtraTrees

## License and credit

QUANT is © Angus Dempster, Daniel F. Schmidt and Geoffrey I. Webb, released under GPL-3.0 (see `original/LICENSE`). `improvedQuant/` is a modified copy of their code; this repository is also released under GPL-3.0.
Datasets: UCR time series classification archive (Dau et al., 2019) and the bake off redux (Middlehurst, Schäfer & Bagnall, 2024).
