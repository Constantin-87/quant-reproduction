# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# QUANT: A Minimalist Interval Method for Time Series Classification
# https://arxiv.org/abs/2308.00928
# (update to handle multivariate per https://github.com/angus924/aaltd2024)

import numpy as np

import torch, torch.nn.functional as F

# == generate intervals ========================================================

def make_intervals(input_length, depth, depths=None, interval_types="both"):
    """Create intervals at selected zero-based depths.

    `depth` preserves original semantics: depth=6 includes levels 0..5.
    `depths` optionally selects explicit levels. interval_types:
    both, normal, shift, or alternating (even=normal, odd=shift).
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

        can_shift = n > 1 and intervals_n.diff().median() > 1
        if use_shift and can_shift:
            shift = int(np.ceil(input_length / n / 2))
            intervals.append(intervals_n[:-1] + shift)

    if not intervals:
        raise ValueError("No intervals generated; check depth and selected levels.")
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

# == interval model (per representation) =======================================

class IntervalModel():

    def __init__(self, input_length, depth = 6, div = 4, depths=None, interval_types="both"):

        assert div >= 1
        assert depth >= 1

        self.div = div

        self.intervals = \
        make_intervals(
            input_length = input_length,
            depth        = depth,
            depths       = depths,
            interval_types = interval_types,
        )

    def fit(self, X, Y):

        pass

    def transform(self, X):

        features = []

        for a, b in self.intervals:

            features.append(
                f_quantile(X[..., a:b], div = self.div).squeeze(1)
            )
        
        return torch.cat(features, -1)

    def fit_transform(self, X, Y):

        self.fit(X, Y)
        
        return self.transform(X)
    
# == quant =====================================================================

class Quant():

    def __init__(self, depth = 6, div = 4, depths=None, interval_types="both"):

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div
        self.depths = depths
        self.interval_types = interval_types

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
                interval_types = self.interval_types
            )

            features.append(
                self.models[index].fit_transform(Z, Y)
            )
        
        self.fitted = True
        
        return torch.cat(features, -1)