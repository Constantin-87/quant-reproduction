# QUANT: our results vs the paper

Written by `notebooks/01_reproduction.ipynb` (section 10). 142 datasets x 30 resamples (resample 0 = the
original UCR split, 1-29 = the bake-off redux splits), default settings (depth 6, div 4; ExtraTrees with 200 trees),
one CPU core; v0 has no seed, v1 uses seed 0.

|  | Paper | Published results file | v0: the authors' code, our run | v1: with timers, our run |
|---|---|---|---|---|
| Mean accuracy, all 142 datasets | 0.8568 | 0.8568 | 0.8552 | 0.8548 |
| Mean accuracy, the 112 original datasets | 0.8671 | 0.8671 | 0.8672 | 0.8669 |
| Mean accuracy, the 138 unchanged datasets | - | 0.8546 | 0.8548 | 0.8545 |
| Datasets within noise of the published results (138 unchanged) | - | - | 138 of 138 | 137 of 138 |
| Time for all 142 datasets on one CPU core (minutes per resample) | under 15 | 14.86 | - (no timers) | 12.2 (12.1 to 12.3) |

- 4 datasets (Covid3MonthDiscrete, FloodModeling1Discrete, FloodModeling2Discrete, FloodModeling3Discrete) were corrected after the paper; the published results were computed on
  the old versions, so the method is compared on the other 138.
- Within noise: |z| < 2, where z is the difference of the per-dataset means over the 30 resamples divided by its
  expected noise. Datasets significant after the Holm correction: v0 0, v1 0.
- Time: QUANT transform, fit and predict, data loading excluded, as in the paper. The published timings come from
  the authors' cluster nodes (Intel Xeon), ours from an Intel Core Ultra 7 155H laptop.
