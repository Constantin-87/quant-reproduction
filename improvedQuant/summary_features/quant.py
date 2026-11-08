# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# QUANT: A Minimalist Interval Method for Time Series Classification
# https://arxiv.org/abs/2308.00928
# (update to handle multivariate per https://github.com/angus924/aaltd2024)

import numpy as np

import torch, torch.nn.functional as F

# == generate intervals ========================================================

def make_intervals(input_length, depth, depths=None, interval_types="both", return_levels=False):
    """Create intervals at selected zero-based depths.

    `depth` preserves original semantics: depth=6 includes levels 0..5.
    `depths` optionally selects explicit levels. interval_types:
    both, normal, shift, or alternating (even=normal, odd=shift).
    With return_levels=True, also returns the depth level of every interval.
    """
    max_level = min(depth - 1, int(np.log2(input_length)))
    selected_depths = list(range(max_level + 1)) if depths is None else list(depths)
    if not selected_depths:
        raise ValueError("At least one depth must be selected.")
    if any(level < 0 for level in selected_depths):
        raise ValueError("Depth values must be non-negative.")
    if interval_types not in {"both", "normal", "shift", "alternating"}:
        raise ValueError("interval_types must be both, normal, shift, or alternating.")

    intervals = []
    levels = []
    for level in selected_depths:
        if level > max_level:
            continue
        n = 2 ** level
        indices = torch.linspace(0, input_length, n + 1).long()
        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        use_normal = interval_types in {"both", "normal"} or (
            interval_types == "alternating" and level % 2 == 0
        )
        use_shift = interval_types in {"both", "shift"} or (
            interval_types == "alternating" and level % 2 == 1
        )
        if use_normal:
            intervals.append(intervals_n)
            levels += [level] * len(intervals_n)

        can_shift = n > 1 and intervals_n.diff().median() > 1
        if use_shift and can_shift:
            shift = int(np.ceil(input_length / n / 2))
            intervals.append(intervals_n[:-1] + shift)
            levels += [level] * (len(intervals_n) - 1)

    if not intervals:
        raise ValueError("No intervals generated; check depth and selected levels.")
    if return_levels:
        return torch.cat(intervals), levels
    return torch.cat(intervals)

# == quantile function =========================================================

def f_quantile(X, div = 4):

    n = X.shape[-1]

    if n == 1:

        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])
    
    else:
        
        num_quantiles = 1 + (n - 1) // div

        if num_quantiles == 1:

            quantiles = X.quantile(torch.tensor([0.5]), dim = -1).permute(1, 2, 0)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])
        
        else:
            
            quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim = -1).permute(1, 2, 0)
            quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims = True)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# == summary statistics function ===============================================

# order-free statistics (describe the distribution of values in the interval)
#   mean, std, min, max : the usual
#   nmp   : normalised mean position, (mean - min) / (max - min); 0.5 for uniform values
#   skew  : skewness (direction of asymmetry)
#   kurt  : excess kurtosis (tail weight; off by default, mostly redundant with skew/min/max)
# temporal statistics (depend on the order of the values in the interval)
#   slope   : least-squares slope of the values against time (per time step)
#   argmax  : position of the maximum within the interval, in [0, 1]
#   argmin  : position of the minimum within the interval, in [0, 1]
#   mac     : mean absolute change between consecutive values (roughness)
#   mcr     : mean-crossing rate (fraction of consecutive pairs that cross the mean)
STAT_NAMES = ("mean", "std", "min", "max", "nmp", "skew", "kurt", "slope", "argmax", "argmin", "mac", "mcr")
DEFAULT_STATS = ("mean", "std", "min", "max", "nmp", "skew", "slope", "argmax", "argmin")

_EPS = 1e-8

def check_stats(stats):
    stats = list(DEFAULT_STATS if stats is None else stats)
    if not stats:
        raise ValueError("At least one statistic must be selected.")
    unknown = [s for s in stats if s not in STAT_NAMES]
    if unknown:
        raise ValueError(f"Unknown statistics {unknown}; choose from {list(STAT_NAMES)}.")
    return list(dict.fromkeys(stats))

def f_stats(X, stats = DEFAULT_STATS):
    """Summary statistics of each interval; same output layout as f_quantile: (N, 1, C * len(stats))."""

    n = X.shape[-1]

    mean = X.mean(-1)
    d = X - mean.unsqueeze(-1)

    needs_var = {"std", "skew", "kurt"}.intersection(stats)
    if needs_var:
        var = (d * d).mean(-1)
        std = var.sqrt()

    if {"min", "max", "nmp"}.intersection(stats):
        mn = X.amin(-1)
        mx = X.amax(-1)

    out = []

    for s in stats:

        if s == "mean":
            out.append(mean)

        elif s == "std":
            out.append(std)

        elif s == "min":
            out.append(mn)

        elif s == "max":
            out.append(mx)

        elif s == "nmp":
            rng = mx - mn
            out.append(torch.where(rng > _EPS, (mean - mn) / rng.clamp_min(_EPS), torch.full_like(rng, 0.5)))

        elif s == "skew":
            m3 = (d ** 3).mean(-1)
            out.append(torch.where(std > _EPS, m3 / std.clamp_min(_EPS) ** 3, torch.zeros_like(std)))

        elif s == "kurt":
            m4 = (d ** 4).mean(-1)
            out.append(torch.where(var > _EPS, m4 / var.clamp_min(_EPS) ** 2 - 3, torch.zeros_like(var)))

        elif s == "slope":
            t = torch.arange(n, dtype = X.dtype, device = X.device) - (n - 1) / 2
            denom = (t * t).sum().clamp_min(_EPS)
            out.append((d * t).sum(-1) / denom)

        elif s == "argmax":
            out.append(X.argmax(-1).to(X.dtype) / max(n - 1, 1))

        elif s == "argmin":
            out.append(X.argmin(-1).to(X.dtype) / max(n - 1, 1))

        elif s == "mac":
            out.append(X.diff().abs().mean(-1) if n > 1 else torch.zeros_like(mean))

        elif s == "mcr":
            out.append(((d[..., 1:] * d[..., :-1]) < 0).to(X.dtype).mean(-1) if n > 1 else torch.zeros_like(mean))

    features = torch.nan_to_num(torch.stack(out, -1))   # (N, C, k)

    return features.reshape(features.shape[0], 1, features.shape[1] * features.shape[2])

# == interval model (per representation) =======================================

class IntervalModel():

    def __init__(self, input_length, depth = 6, div = 4, depths=None, interval_types="both",
                 interval_features = "quantile", stats = None, stats_ratio = 2):

        assert div >= 1
        assert depth >= 1
        assert interval_features in {"quantile", "stats", "auto"}, "interval_features must be quantile, stats or auto."
        assert stats_ratio > 0

        self.div = div
        self.stats = check_stats(stats)

        self.intervals, levels = \
        make_intervals(
            input_length = input_length,
            depth        = depth,
            depths       = depths,
            interval_types = interval_types,
            return_levels = True,
        )

        # one decision per depth level (so every interval in a level has the same feature layout):
        #   quantile : quantiles everywhere (original QUANT, the default)
        #   stats    : summary statistics everywhere
        #   auto     : summary statistics at a level when its median interval would give at least
        #              stats_ratio * len(stats) quantiles, otherwise quantiles
        lengths = (self.intervals[:, 1] - self.intervals[:, 0]).tolist()
        levels = np.asarray(levels)
        self.level_uses_stats = {}
        for level in np.unique(levels):
            level_lengths = np.asarray(lengths)[levels == level]
            n_quantiles = 1 + (int(np.median(level_lengths)) - 1) // div
            self.level_uses_stats[int(level)] = (
                interval_features == "stats"
                or (interval_features == "auto" and n_quantiles >= stats_ratio * len(self.stats))
            )
        self.interval_levels = [int(level) for level in levels]

    def fit(self, X, Y):

        pass

    def transform(self, X):

        features = []

        for (a, b), level in zip(self.intervals, self.interval_levels):

            if self.level_uses_stats[level]:
                features.append(f_stats(X[..., a:b], stats = self.stats).squeeze(1))
            else:
                features.append(f_quantile(X[..., a:b], div = self.div).squeeze(1))
        
        return torch.cat(features, -1)

    def fit_transform(self, X, Y):

        self.fit(X, Y)
        
        return self.transform(X)
    
# == quant =====================================================================

class Quant():

    def __init__(self, depth = 6, div = 4, depths=None, interval_types="both",
                 interval_features = "quantile", stats = None, stats_ratio = 2):

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div
        self.depths = depths
        self.interval_types = interval_types
        self.interval_features = interval_features
        self.stats = stats
        self.stats_ratio = stats_ratio

        self.representation_functions = \
        (
            lambda X : X,
            lambda X : F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X : X.diff(n = 2),
            lambda X : torch.fft.rfft(X).abs(),
        )

        self.models = {}

        self.fitted = False

    def transform(self, X):

        assert self.fitted, "not fitted"

        features = []

        for index, function in enumerate(self.representation_functions):

            Z = function(X)

            features.append(
                self.models[index].transform(Z)
            )
        
        return torch.cat(features, -1)
    
    def fit_transform(self, X, Y):

        features = []

        for index, function in enumerate(self.representation_functions):

            Z = function(X)

            self.models[index] = \
            IntervalModel(
                input_length = Z.shape[-1],
                depth        = self.depth,
                div          = self.div,
                depths       = self.depths,
                interval_types = self.interval_types,
                interval_features = self.interval_features,
                stats        = self.stats,
                stats_ratio  = self.stats_ratio,
            )

            features.append(
                self.models[index].fit_transform(Z, Y)
            )
        
        self.fitted = True
        
        return torch.cat(features, -1)