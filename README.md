# Reproducing and Improving QUANT

Reproduction study of

> Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). **QUANT: A Minimalist Interval Method for Time Series Classification.** *Data Mining and Knowledge Discovery.* [doi:10.1007/s10618-024-01036-9](https://doi.org/10.1007/s10618-024-01036-9) · [arXiv:2308.00928](https://arxiv.org/abs/2308.00928) · [original code](https://github.com/angus924/quant)

COMP41850 AI for Time Series, group project. Report: *TODO link*.

This repository holds everything behind the report: the reproduction of QUANT with the authors' code (v0) and an
instrumented copy (v1), and the improvements we tried, each with its code, its results files and the notebook that
analyses it: other classifiers on QUANT's features and a cheaper TabPFN (v2, v3), a supervised feature extractor for
TabPFN (v4), interval depths, summary statistics instead of quantiles, and weighted interval sampling. Every results
file can be made again with its command in [`experiments.md`](experiments.md); the results themselves are discussed
in the report.


## Repository structure

```
original/                       the authors' code and published results, unmodified (GPL-3.0)
improvedQuant/
  v1_classifiers/               v1: QUANT with timers, metrics, a seed and a choice of classifier
  features/                     v2, v3, v4: TabPFN's cost settings (v2), the lean extractor for TabPFN quant_tabpfn.py
                                (v3) and its supervised extractor (v4), all run by run.py
  baselines/                    the simple baselines: majority class, raw values + ExtraTrees, catch22 + random forest,
                                raw values + TabPFN
  depth_selection/              QUANT with a choice of interval depths and interval types
  summary_features/             QUANT with summary statistics instead of (some) quantiles
  weighted/                     QUANT with weighted interval sampling (copies of the larger intervals' features)
  dataSetLists/                 the dataset lists (ucr142 = a copy of original/code/ucr142.txt; smoke-small/medium = the
                                development lists; worst3 = the datasets where v3 loses most to QUANT + ExtraTrees;
                                dev34 = the 34 datasets used to develop v3 and v4, heldout108 = the other 108;
                                dataset_types.csv = the archive type of every dataset, for the depth analysis)
notebooks/                      the analyses (01 to 06, depths_analysis, summary_features), common.py (shared names,
                                colours and loading) and cd_diagram.py (critical-difference diagrams, Fawaz et al.,
                                used by depths_analysis and summary_features)
results/                        every results file of our runs, the comparison with the paper, and in bakeoff-redux/
                                the published accuracies and times of 11 classifiers (downloaded, see experiments.md)
figures/                        the figures of the report (written by notebooks 01 and 05; cd_extratrees_* and
                                accuracy_vs_time_extratrees by depths_analysis and summary_features)
run.bat, run_depth_assess.bat,  the batch files of the depth and summary-statistics runs (written before the
run_stats.bat                   runners moved and the files were renamed; the current commands are in experiments.md)
Ideas.md                        a list of experiment ideas
data/                           datasets and resample indices (not in git, see section 2)
experiments.md                  every version (v0, v1, v2, v3, v4), its notebook and the command for every file in results/
requirements-gpu.txt            packages of .venv-gpu: the other runs (with XGBoost, CatBoost, TabPFN and aeon)
requirements-original-lock.txt  exact package versions of the reproduction environment (.venv)
requirements.txt                packages of .venv: the reproduction runs and the notebooks
```

`quant.py` in `v1_classifiers/` and `features/` is the authors' code, unchanged.


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

**Preparation**

- Keep only the 142 names in `original/code/ucr142.txt`: remove the 16 originals with unequal lengths or missing
  values (QUANT uses their fixed versions), Fungi, and the resampled `.ts` copies in the index download.
- Rename `_eq` / `_nmv` / `_disc` to `Eq` / `Nmv` / `Discrete` (24 dataset and 24 index folders).
- Layout: `data/ts/<name>/<name>_TRAIN.ts` and `_TEST.ts`; `data/indices/<name>/resample<i>Indices_TRAIN.txt`.
- The downloads have the corrected versions of 4 datasets (Covid3MonthDiscrete, FloodModeling1-3Discrete), while the
  published results were computed on the old ones (notebook 01, section 2).


## 3. Reproduce the results

Every version (v0, v1, v2, v3, v4), what it is, the notebook that analyses it and the command for each of its results
files are in [`experiments.md`](experiments.md). The two reproduction runs take about 6 hours each on one CPU core,
the full runs of v3 and v4 about 3 and 4.7 hours (all cores + GPU), the ablation of v4 about 2 hours, the CPU
baselines about an hour, raw values + TabPFN about 13.6 hours, the five seeds on the default split about 70 minutes; the smoke tests a few minutes to about an hour
each, each depth or summary-statistics configuration under 2 hours (all CPU cores) and each of the four
weighted-sampling settings a few hours. The selection log of v4's full run (`results/*_ucr142_*_selection.csv`,
about 200 MB) is not in the repository; running the command again writes it, and the repository has the summary
notebook 04 makes from it. Earlier development steps (other feature families, the step-by-step build of the supervised extractor) are
not kept; their results are in the git history.


## 4. Notebooks

| Notebook | Question | Results files |
|---|---|---|
| `01_reproduction` | Published vs v0 vs v1 on the published setup (142 datasets x 30 resamples, one CPU core): does v0 reproduce the published results, does v1 leave the method unchanged, and does the single-core timing claim hold? | `v0_ucr142_onecore-cpu`, `v1_ucr142_onecore-cpu` |
| `02_classifiers` | The four classifiers on QUANT's features (ExtraTrees, XGBoost, CatBoost, TabPFN), small vs medium datasets: accuracy, probability quality and time | `v1_smoke-{small,medium}-r0_allcores-gpu` |
| `03_cheap-tabpfn` | A cheaper TabPFN: fewer passes and features and the model cache (v2), and the lean extractor `quant_tabpfn.py` (v3), against QUANT + ExtraTrees and TabPFN's default | `v2_*` and `v3_*` on the smoke lists, `v1_smoke-{small,medium}_allcores-gpu` |
| `04_supervised-extractor` | Why v3 loses on its 3 worst datasets; the supervised extractor (v4) and its ablation (each part switched off, on the 34 development datasets, Holm-corrected); what v4 keeps and why; which lag-d views it uses | the `worst3` files, `v4-abl-*`, v4's selection summary, and from the full runs `v1_ucr142_allcores-cpu`, `v3_tabpfn-e2f200-both_ucr142_allcores-gpu`, `v4_tabpfn-e2f200_ucr142_allcores-gpu` |
| `05_full-benchmark` | v3 and v4 on the published setup (142 datasets x 30 resamples): the protocol and its threats; mean accuracy; per dataset vs QUANT + ExtraTrees with Holm-corrected tests; simple baselines; the published bake-off redux classifiers (Friedman's test, critical-difference diagram, v4 against every other); the 34 development vs 108 held-out datasets; the default split alone; confidence intervals and the spread over resamples; five seeds; balanced accuracy, macro F1, AUROC, log loss, Brier score; confusion matrices; time by phase and memory; accuracy against single-core time; the report figures | `v3_tabpfn-e2f200-both_ucr142_allcores-gpu`, `v4_tabpfn-e2f200_ucr142_allcores-gpu`, `v1_ucr142_onecore-cpu`, `v1_ucr142_allcores-cpu`, `baselines_ucr142_allcores-cpu`, `baselines-tabpfn_ucr142_allcores-gpu`, `v4-seeds_*`, `v1-seeds_*`, `v4_tabpfn-e2f200_ucr142-r0_onecore-cpu`, `bakeoff-redux/*` |
| `06_weighted-sampling` | Does drawing the features of larger intervals more often (copies) help ExtraTrees, at depth 6 or 8 with m/4 or m/16 quantiles, against QUANT and a control with as many candidate features per split? | `w-d{6,8}v{4,16}_ucr142` |
| `07_depths_analysis` | Which interval depths QUANT needs (depths removed, odd or even depths, alternating normal/shifted intervals), accuracy against features and time, per dataset size and type, critical-difference diagrams, assessment of indiviual depth importance | `depths_1_6_baseline`, `no_depth_*`, `odd_depths`, `even_depths`, `alternating_depths` |
| `08_summary_features` | Summary statistics (mean, spread, slope, ...) instead of quantiles at all or some depths | `basic_baseline`, `all_stats`, `half_stats`, `equal_stats`, `double_stats` |


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
The published accuracies and fit and predict times of the other classifiers in `results/bakeoff-redux/` are the bake
off redux's reference results, downloaded unchanged from
https://timeseriesclassification.com/results/ReferenceResults/classification/ (`accuracy/`, `fittime/`, `predicttime/`).
