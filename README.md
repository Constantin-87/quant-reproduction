# Reproducing and Improving QUANT

Reproduction study of

> Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). **QUANT: A Minimalist Interval Method for Time Series Classification.** *Data Mining and Knowledge Discovery.* [doi:10.1007/s10618-024-01036-9](https://doi.org/10.1007/s10618-024-01036-9) · [arXiv:2308.00928](https://arxiv.org/abs/2308.00928) · [original code](https://github.com/angus924/quant)

COMP41850 AI for Time Series, group project. Report: *TODO link*.

**In short**
- **Reproduction:** the published accuracy reproduces on the 138 datasets that were not changed after the paper
  (0.8548 vs 0.8546). The gap on all 142 datasets (0.8552 vs 0.8568) comes from 4 datasets that were corrected after
  the paper. The timing claim holds: 12.2 minutes for all 142 datasets on one CPU core (paper: under 15).
- **Classifier:** TabPFN on QUANT's features is the most accurate of four classifiers (ExtraTrees, XGBoost,
  CatBoost, TabPFN), but about 30 times slower than ExtraTrees. Extra feature sets (lag-d differences, MultiRocket,
  RDST, MR-Hydra) help only on tiny training sets.
- **v4, a cheap TabPFN** (a lean extractor, 200 features picked per dataset, 2 passes, the model built once): on all
  142 datasets x 30 resamples 0.8733 against 0.8548 for QUANT + ExtraTrees, better on 111 of 142 datasets
  (Wilcoxon p = 2e-12), in 3.0 hours against 1.7 on all CPU cores.
- **v9, a supervised extractor** (candidates from QUANT's views with a fifth of QUANT's quantiles, lag-d views chosen
  per dataset, catch22 and pattern templates; at most 200 kept): 0.8849, better than QUANT + ExtraTrees on 126 of 142
  datasets (p = 2e-21) and than v4 on 99 (p = 3e-11), in 4.7 hours. Its predecessor v8 (0.8827, 5.0 hours) and the
  analysis of the datasets where it still loses led to v9's changes (notebook 05).


## Repository structure

```
original/                       the authors' code and published results, unmodified (GPL-3.0)
improvedQuant/
  v1_classifiers/               v1: QUANT with timers, metrics, a seed and a choice of classifier
  features/                     v2-v9: QUANT + extra features (v2), TabPFN's cost settings (v3), the lean extractor
                                for TabPFN quant_tabpfn.py (v4) and its supervised extractor (v5-v9); feature_test.py:
                                the tests of which of QUANT's features help TabPFN (v9)
  dataSetLists/                 the dataset lists (ucr142 = a copy of original/code/ucr142.txt, smoke-small/medium/large,
                                worst3/worst7/worst10 = the datasets where a version loses most to QUANT + ExtraTrees,
                                deep6, chlorine-uwave: the tests behind v9)
notebooks/                      the analyses (01 to 06) and common.py (shared names, colours and loading)
results/                        every results file of our runs, and the comparison with the paper
data/                           datasets and resample indices (not in git, see section 2)
experiments.md                  every version (v0-v9), its notebook and the command for every file in results/
requirements-gpu.txt            packages of .venv-gpu: the other runs (with XGBoost, CatBoost, TabPFN and aeon)
requirements-original-lock.txt  exact package versions of the reproduction environment (.venv)
requirements.txt                packages of .venv: the reproduction runs and the notebooks
```

`quant.py` is the authors' file, unchanged (byte for byte) in every folder of `improvedQuant/`.


## 1. Environment

Python 3.12, two environments:

**`.venv`, for the reproduction runs and the notebooks** (CPU build of PyTorch):

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

**`.venv-gpu`, for everything else** (CUDA build of PyTorch, for TabPFN, XGBoost and CatBoost on the GPU; aeon for
the feature versions):

```powershell
uv venv --python 3.12 .venv-gpu
.venv-gpu\Scripts\Activate.ps1
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements-gpu.txt
```

TabPFN's model weights need a Prior Labs account, the accepted licence and an API key stored locally
(`~/.cache/tabpfn/auth_token`, or the `TABPFN_TOKEN` environment variable).

What we used: Windows 11 laptop, Intel Core Ultra 7 155H (22 threads), NVIDIA RTX 4050 Laptop GPU (6 GB);
Python 3.12.10, PyTorch 2.14.0 (CPU, and CUDA 13.0 in `.venv-gpu`), scikit-learn 1.9.1, TabPFN 9.0.0, XGBoost 3.4.1,
CatBoost 1.2.10, aeon 1.6.0.

## 2. Data

The paper names no download files; the links are from the
[bake-off notebook](https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/publications/y2023/tsc_bakeoff/tsc_bakeoff_2023.ipynb)
referenced in `original/doc.md`.

**Datasets**

- 112 UCR datasets ("Univariate aeon formatted ts files"): https://www.timeseriesclassification.com/dataset.php
- 30 new bake-off datasets: https://drive.google.com/file/d/1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3/view

**Resample indices** (not in the authors' repository; resample 0 is the original train/test split)

- 112 UCR datasets (Java tsml indices): https://drive.google.com/file/d/1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3/view
- 30 new bake-off datasets (Python tsml-eval indices): https://drive.google.com/file/d/1aLBP_nhnoqz075puKg30zuF3F_QBOXYM/view


## 3. Reproduce the results

Every version (v0 to v9), what it is, the notebook that analyses it and the command for each of its results files are
in [`experiments.md`](experiments.md). The two reproduction runs take about 6 hours each on one CPU core, the full
runs of v4, v8 and v9 about 3.5, 5 and 4.7 hours (all cores + GPU); the smoke tests a few minutes to about an hour
each. The selection logs of the full runs of v8 and v9 (`results/*_ucr142_*_selection.csv`, about 180-200 MB each)
are not in the repository; running the command again writes them. Results of experiments that are not in the report
are kept locally in `results_dropped/`, which is not in the repository.


## 4. Notebooks

| Notebook | Question | Results files |
|---|---|---|
| `01_reproduction` | Published vs v0 vs v1 on the published setup (142 datasets x 30 resamples, one CPU core): does v0 reproduce the published results, does v1 leave the method unchanged, and does the single-core timing claim hold? | `v0_ucr142_onecore-cpu`, `v1_ucr142_onecore-cpu` |
| `02_classifiers` | The four classifiers on QUANT's features: one core vs all cores, with and without the GPU, small vs medium datasets | `v1_smoke-{small,medium}-r0_*` |
| `03_more-features` | v2: QUANT + extra features (lag-d differences, MultiRocket, RDST, MR-Hydra) for all four classifiers, on the small and the medium datasets | `v1_smoke-{small,medium}_allcores-gpu`, `v2-*` |
| `04_cheap-tabpfn` | A cheaper TabPFN: its cost settings and the model cache (v3), a lean extractor for TabPFN (`quant_tabpfn.py`), fewer features and three ways to pick them (v4) | `v3-*`, `v4-*`, `v4_*` on the smoke lists, and the v1 and v2-lagd files of notebook 03 |
| `05_supervised-extractor` | Why v4 loses on its 3 worst datasets, and a supervised extractor for TabPFN (v5 to v9: candidates from QUANT's views, lag-d views, catch22 and pattern templates; relevance check, ExtraTrees importance and a greedy selection without near-copies); accuracy and time on the small, medium and 3 worst datasets; which features are kept or dropped and why; v8 on all 142 datasets and why it still loses on 10 of them; from v8 to v9: where the time goes, which of QUANT's features TabPFN needs, 200 features | `v5-*` to `v8-*` on the smoke lists and `worst3` and their `_selection.csv`, the `worst3` and `worst10` files, `v8_tabpfn-e2f250_ucr142_allcores-gpu`, `v8-depth5_*`, `v9-ablation_*`, `v9-quantiles_*`, `v9-anova_*`, `v3-quant_tabpfn-e2f200-both_*`, and from the full runs `v1_ucr142_allcores-cpu`, `v4_tabpfn-e2f200-both_ucr142_allcores-gpu`, `v9_tabpfn-e2f200_ucr142_allcores-gpu` |
| `06_full-benchmark` | v4 and v9 on the published setup (142 datasets x 30 resamples) vs QUANT + ExtraTrees: mean accuracy and rank, per dataset with significance tests, time per run | `v4_tabpfn-e2f200-both_ucr142_allcores-gpu`, `v9_tabpfn-e2f200_ucr142_allcores-gpu`, `v1_ucr142_onecore-cpu`, `v1_ucr142_allcores-cpu` |


## 5. Results compared with the paper

The paper's claims next to the published results file and our two runs of the original method (v0: the authors'
code; v1: with timers), on the published setup: [`results/comparison_with_paper.md`](results/comparison_with_paper.md).
Notebook 01 writes this table from the same results it analyses.


## Notes on the original `run.py`

- It uses all CPU cores, the four `set` variables in `experiments.md` can be used to limit it
  to one core without changing the code.
- It does not seed ExtraTrees, so accuracies differ slightly from run to run and from the published values.
- It writes `results.csv` into the `-o` folder, only at the very end, and overwrites an existing one.
- It re-reads the `.ts` files for every resample.
- A dataset list with only one name makes it crash; use at least two.


## License and credit

QUANT is © Angus Dempster, Daniel F. Schmidt and Geoffrey I. Webb, released under GPL-3.0 (see `original/LICENSE`). `improvedQuant/` is a modified copy of their code; this repository is also released under GPL-3.0.
Datasets: UCR time series classification archive (Dau et al., 2019) and the bake off redux (Middlehurst, Schäfer & Bagnall, 2024).
