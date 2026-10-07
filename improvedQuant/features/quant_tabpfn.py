# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# QUANT: A Minimalist Interval Method for Time Series Classification
# https://arxiv.org/abs/2308.00928
# (update to handle multivariate per https://github.com/angus924/aaltd2024)

# quant_tabpfn.py: a copy of the authors' quant.py, changed for TabPFN with a small feature budget (1-2 passes, at
# most 300 features), where near-copies of features waste the budget. The authors' lines are kept, commented out
# with "# ORIGINAL:", next to each change:
#   1. fewer levels: depth 5 instead of 6 (the smallest pieces, mostly noise, are dropped)
#   2. fewer quantiles per piece: div 8 instead of 4 (fewer, more distinct numbers)
#   3. no shifted pieces (the half-overlapping copies of every level)
#   4. no subtraction of the piece's average (a trick for trees, which look at one feature at a time)
#   5. per piece also its mean, spread and trend (more information per feature than more quantiles)
#   6. optional (num_quantiles > 0): a fixed number of quantiles per piece instead of one per div points, e.g. 2 =
#      min and max; with depth 4 and lags 4, 16 about 450 features (v4; run.py: --quant tabpfn -m 4 --quantiles 2)
#   7. SupervisedExtractor (at the end of this file; v5 to v9, run.py: --quant supervised): candidates from several
#      views (+ optionally catch22 and pattern templates), a relevance check and a greedy selection of at most
#      `budget` non-redundant features, all decided on the training labels
#   8. optional (fraction > 0): a fraction of the authors' quantiles (one per 4 points), odd so that the median stays
#      in, e.g. 0.2 = 25 instead of 125 for a piece of 500 points (v9: QUANT's four views of the SupervisedExtractor,
#      run.py: --quantile_fraction 0.2)

import numpy as np

import torch, torch.nn.functional as F

# == generate intervals ========================================================

def make_intervals(input_length, depth):

    exponent = \
    min(
        depth,
        int(np.log2(input_length)) + 1
    )

    intervals = []

    for n in 2 ** torch.arange(exponent):

        indices = torch.linspace(0, input_length, n + 1).long()

        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        intervals.append(intervals_n)

        # change 3: no shifted pieces
        # ORIGINAL:
        # if n > 1 and intervals_n.diff().median() > 1:
        #
        #     shift = int(np.ceil(input_length / n / 2))
        #
        #     intervals.append((intervals_n[:-1] + shift))

    return torch.cat(intervals)

# == quantile function =========================================================

def quantile_count(n, div, num_quantiles = 0, fraction = 0):
    """The number of quantiles of a piece of n > 1 points (change 6 and 8; the authors: 1 + (n - 1) // div)."""
    if fraction > 0:                                    # change 8: a fraction of the authors' count, odd (median in)
        k = max(1, round((1 + (n - 1) // 4) * fraction))
        return min(k + 1 - k % 2, n)
    return min(num_quantiles, n) if num_quantiles > 0 else 1 + (n - 1) // div

# change 2: div 8 instead of 4
# ORIGINAL: def f_quantile(X, div = 4):
def f_quantile(X, div = 8, num_quantiles = 0, fraction = 0):

    n = X.shape[-1]

    if n == 1:

        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])

    else:

        # change 6 and 8: a fixed number or a fraction of the quantiles per piece (0 = one per div points, as the authors)
        # ORIGINAL: num_quantiles = 1 + (n - 1) // div
        num_quantiles = quantile_count(n, div, num_quantiles, fraction)

        if num_quantiles == 1:

            quantiles = X.quantile(torch.tensor([0.5]), dim = -1).permute(1, 2, 0)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

        else:

            quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim = -1).permute(1, 2, 0)
            # change 4: no subtraction of the piece's average
            # ORIGINAL: quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims = True)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# change 5: three summaries of a piece, next to its quantiles (min and max are already quantiles 0 and 1)
def f_summary(X):
    """Mean, spread (standard deviation) and trend (slope of a straight line through the piece) of every piece."""
    n = X.shape[-1]
    mean = X.mean(-1)
    if n == 1:                                          # a piece of one value: no spread, no trend
        return torch.cat((mean, torch.zeros_like(mean), torch.zeros_like(mean)), -1)
    std = X.std(-1)
    t = torch.arange(n, dtype = X.dtype) - (n - 1) / 2  # the positions in the piece, centred on 0
    slope = (X * t).sum(-1) / (t * t).sum()             # least-squares slope against the position
    return torch.cat((mean, std, slope), -1)

# == interval model (per representation) =======================================

class IntervalModel():

    # change 1 and 2: depth 5 and div 8 instead of 6 and 4
    # ORIGINAL: def __init__(self, input_length, depth = 6, div = 4):
    def __init__(self, input_length, depth = 5, div = 8, num_quantiles = 0, fraction = 0):

        assert div >= 1
        assert depth >= 1

        self.div = div
        self.num_quantiles = num_quantiles                  # change 6
        self.fraction = fraction                            # change 8

        self.intervals = \
        make_intervals(
            input_length = input_length,
            depth        = depth,
        )

    def fit(self, X, Y):

        pass

    def transform(self, X):

        features = []

        for a, b in self.intervals:

            features.append(
                f_quantile(X[..., a:b], div = self.div, num_quantiles = self.num_quantiles,
                           fraction = self.fraction).squeeze(1)
            )
            features.append(f_summary(X[..., a:b]))     # change 5

        return torch.cat(features, -1)

    def fit_transform(self, X, Y):

        self.fit(X, Y)

        return self.transform(X)

# == quant =====================================================================

class Quant():

    # change 1 and 2: depth 5 and div 8 instead of 6 and 4
    # ORIGINAL: def __init__(self, depth = 6, div = 4):
    def __init__(self, depth = 5, div = 8, num_quantiles = 0):

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div
        self.num_quantiles = num_quantiles                  # change 6

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
                num_quantiles = self.num_quantiles,     # change 6
            )

            features.append(
                self.models[index].fit_transform(Z, Y)
            )

        self.fitted = True

        return torch.cat(features, -1)

# == change 7: a supervised extractor ============================================
#
# Instead of returning every feature and leaving the choice to a later filter, it looks at the training labels and
# returns at most `budget` strong, non-redundant features itself:
#   1. candidates: the interval features of several views of the series (QUANT's four, a shape-only view, lag-d
#      differences), optionally the catch22 features (whole-series behaviour) and pattern templates (short windows
#      from training series);
#   2. a relevance check: candidates that do not separate the classes on the training data (ANOVA, p >= 0.05) go;
#   3. greedy selection: down the ANOVA ranking (or the ExtraTrees importance, rank = "both"), a feature is taken
#      only if it is not a near-copy (|correlation| >= 0.95) of one already taken, up to the budget.
# Everything is decided in fit_transform on the training data; transform computes the same candidates for new data
# (only the views and catch22 features with kept features) and returns the chosen columns. selection_log records
# what happened to every candidate (notebook 05).
# fill = True makes the limits relative to the dataset: the selection goes on down the ranking past p >= 0.05
# ("kept (weak)") until the budget is full, and when even that is not enough (few candidates, many near-copies) the
# pieces are made finer (depth + 1, up to max_depth) and the selection is redone; catch22 and the templates do not
# depend on the piece size, so they are computed once and shared by these rounds.

def anova(features, Y):
    """ANOVA F-scores and p-values of the columns (constant columns: F = 0, p = 1)."""
    import warnings
    from sklearn.feature_selection import f_classif
    with warnings.catch_warnings(), np.errstate(all = "ignore"):
        warnings.simplefilter("ignore")
        scores, p_values = f_classif(features, Y)
    return np.nan_to_num(scores, nan = 0.0), np.nan_to_num(p_values, nan = 1.0)

def template_distances(S, T, batch = 256):
    """Smallest mean squared distance between each template (rows of T, length w) and any window of each series
    (rows of S): (cases, templates). Uses (x - t)^2 = x^2 - 2 x t + t^2, so no window is copied."""
    w = T.shape[-1]
    out = []
    for start in range(0, S.shape[0], batch):           # in batches of cases, to bound the memory
        x = S[start:start + batch].unsqueeze(1)          # (cases, 1, length)
        cross = F.conv1d(x, T.unsqueeze(1))              # (cases, templates, windows)
        x2 = F.avg_pool1d(x ** 2, w, 1) * w              # (cases, 1, windows): the sum of x^2 per window
        d = (x2 - 2 * cross + (T ** 2).sum(-1)[None, :, None]).clamp(min = 0) / w
        out.append(d.min(-1).values)
    return torch.cat(out)

class SupervisedExtractor():

    QUANT_VIEWS = ("series", "change (smoothed)", "change of change", "rhythm (|FFT|)")

    def __init__(self, depth = 4, div = 8, num_quantiles = 2, lags = (), znorm = False, catch22 = False,
                 templates = False, budget = 250, fill = False, max_depth = 6, c22_shorten = False, rank = "anova",
                 quantile_fraction = 0, seed = 0):
        self.depth, self.div, self.num_quantiles = depth, div, num_quantiles
        # change 8 (v9): QUANT's four views get this fraction of the authors' quantiles instead of num_quantiles (the
        # other views keep num_quantiles); 0 = num_quantiles everywhere
        self.quantile_fraction = quantile_fraction
        self.lags = lags if lags == "auto" else list(lags)   # "auto": see lag_list
        self.znorm, self.catch22, self.templates = znorm, catch22, templates
        self.budget, self.seed = budget, seed
        self.fill, self.max_depth = fill, max_depth      # fill the budget, see fit_transform and select
        self.c22_shorten = c22_shorten                   # catch22 on a shortened copy, see c22_factor
        self.rank = rank                                 # the order of the greedy selection, see select
        self.round_cache = None                          # fit_transform: blocks shared by the fill rounds

    def lag_list(self, length):
        """The lag-d views: the given lags, or for lags = "auto" a ladder 2, 4, 8, ... up to a quarter of the series
        length (at most 64); the selection then decides per dataset which of them are used."""
        if self.lags != "auto":
            return self.lags
        return [2 ** k for k in range(1, 7) if 2 ** k <= length // 4]

    def views(self, X):
        """(name, series) of every view the interval features are computed on."""
        views = [("series", X),
                 ("change (smoothed)", F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1)),
                 ("change of change", X.diff(n = 2)),
                 ("rhythm (|FFT|)", torch.fft.rfft(X).abs())]
        if self.znorm:                                   # shape only: every series to mean 0 and spread 1
            spread = X.std(-1, keepdim = True)
            views.append(("shape (z-normalized)", (X - X.mean(-1, keepdim = True)) / torch.where(spread > 0, spread, 1)))
        for d in self.lag_list(X.shape[-1]):
            if X.shape[-1] > d + 1:
                views.append((f"lag-{d}", X[..., d:] - X[..., :-d]))
        return views

    def interval_meta(self, name, model, length):
        """source, kind, level (piece size) and position of every interval feature, in the column order."""
        meta = []
        for a, b in model.intervals:
            a, b = int(a), int(b)
            n = b - a
            level = int(round(np.log2(length / n))) if n > 0 else 0
            centre = (a + b) / 2 / length
            position = "whole" if level == 0 else ("early" if centre < 1 / 3 else "late" if centre > 2 / 3 else "middle")
            n_quantiles = 1 if n == 1 else quantile_count(n, model.div, model.num_quantiles, model.fraction)
            for kind in ["quantile"] * n_quantiles + ["mean", "spread", "trend"]:
                meta.append({"source": name, "kind": kind, "level": f"1/{2 ** level}", "position": position})
        return meta

    def fit_templates(self, S, Y):
        """Pattern templates: windows of training series, the best ones per class (one class against the rest)."""
        n, length = S.shape
        w = max(3, min(64, length // 8))
        rng = np.random.default_rng(self.seed)
        classes = np.unique(Y)
        per_class = max(1, min(3, 60 // len(classes)))  # 2 classes: 3 each; 60 classes: 1 each
        tries = max(per_class, min(20, 200 // len(classes)))   # candidate windows per class
        templates, owners = [], []
        for c in classes:
            rows = np.flatnonzero(Y == c)
            picks = rng.choice(rows, size = tries)
            starts = rng.integers(0, length - w + 1, size = tries)
            candidates = torch.stack([S[r, s:s + w] for r, s in zip(picks, starts)])
            scores, _ = anova(template_distances(S, candidates).numpy(), (Y == c).astype(int))
            for k in np.argsort(-scores)[:per_class]:
                templates.append(candidates[k])
                owners.append(c)
        self.template_set, self.template_owners = torch.stack(templates), owners

    def c22_factor(self, Z, energy = 0.99, min_length = 32, keep_below = 256):
        """c22_shorten: by how much this view can be shortened for catch22, from the training series themselves. Series
        shorter than keep_below points stay as they are (catch22 is fast on them). For longer ones, the mean spectrum
        (|FFT|^2) says how the detail is spread over fast and slow changes; the factor k is the largest that keeps the
        changes holding `energy` of it (the copy keeps changes up to 1/k of the fastest possible) and leaves at least
        min_length points. Smooth series get a large k, noisy or spiky ones k = 1 (not shortened)."""
        if not self.c22_shorten or Z.shape[-1] < keep_below:
            return 1
        power = (torch.fft.rfft(Z[:, 0, :] - Z[:, 0, :].mean(-1, keepdim = True)).abs() ** 2).mean(0)[1:]
        if power.sum() <= 0:
            return 1
        needed = int((power.cumsum(0) / power.sum() < energy).sum()) + 1          # frequencies holding `energy`
        return int(max(1, min(len(power) // needed, Z.shape[-1] // min_length)))

    def c22_input(self, Z, name):
        """The series catch22 is computed on: Z, or with c22_shorten a copy with the mean of every k points."""
        k = self.c22_k[name]
        return (F.avg_pool1d(Z, k) if k > 1 else Z).double().numpy()

    def candidates(self, X, kept_only = False):
        """All candidate features of X (cases, candidates), in a fixed column order. kept_only = True (new data, after
        fit_transform): only the views and catch22 features with kept features are computed; the others stay 0 (they
        are not returned)."""
        views = self.views(X)
        blocks = [self.models[name].transform(Z) if not kept_only or name in self.kept_views
                  else torch.zeros(len(X), self.view_width[name]) for name, Z in views]
        if self.catch22:
            for name, Z in (("series", X), ("change (smoothed)", views[1][1])):
                if not kept_only:
                    if self.round_cache is None or name not in self.round_cache:
                        block = torch.tensor(self.c22[name].transform(self.c22_input(Z, name)), dtype = torch.float32)
                        if self.round_cache is not None:
                            self.round_cache[name] = block
                    blocks.append(self.round_cache[name] if self.round_cache is not None else block)
                    continue
                block = torch.zeros(len(X), 22)
                columns, transformer = self.c22_kept[name]
                if columns:
                    block[:, columns] = torch.tensor(transformer.transform(self.c22_input(Z, name)), dtype = torch.float32)
                blocks.append(block)
        if self.templates:
            if kept_only or self.round_cache is None:
                blocks.append(template_distances(X[:, 0, :], self.template_set))
            else:
                if "templates" not in self.round_cache:
                    self.round_cache["templates"] = template_distances(X[:, 0, :], self.template_set)
                blocks.append(self.round_cache["templates"])
        return torch.nan_to_num(torch.cat(blocks, -1), nan = 0.0, posinf = 0.0, neginf = 0.0)

    def fit_transform(self, X, Y):
        depth = self.depth
        # catch22 and the templates do not depend on the piece size: computed in the first round, kept for the next
        self.round_cache = {}
        while True:                                      # fill: finer pieces until the budget can be filled
            C = self.fit_round(X, Y, depth)
            if not self.fill or len(self.keep) >= self.budget or depth >= self.max_depth:
                break
            depth += 1
        self.depth_used = depth
        self.round_cache = None                          # new data (transform) is always computed
        return C

    def fit_round(self, X, Y, depth):
        """Candidates with pieces down to 1/2**(depth - 1), and the selection on them."""
        from aeon.transformations.collection.feature_based import Catch22
        import aeon.transformations.collection.feature_based._catch22 as c22_names
        views = self.views(X)
        self.models = {name: IntervalModel(Z.shape[-1], depth = depth, div = self.div, num_quantiles = self.num_quantiles,
                                           fraction = self.quantile_fraction if name in self.QUANT_VIEWS else 0)
                       for name, Z in views}
        meta = [m for name, Z in views for m in self.interval_meta(name, self.models[name], Z.shape[-1])]
        if self.catch22:
            self.c22, self.c22_k = {}, {}
            for name, Z in (("series", X), ("change (smoothed)", views[1][1])):
                self.c22_k[name] = self.c22_factor(Z)
                self.c22[name] = Catch22(replace_nans = True).fit(self.c22_input(Z, name), Y)
                meta += [{"source": "catch22", "kind": f"{feature} ({name})", "level": "1/1", "position": "whole"}
                         for feature in c22_names.feature_names]
        if self.templates:
            if "template_set" not in self.round_cache:   # chosen once, the same in every round
                self.fit_templates(X[:, 0, :], Y)
                self.round_cache["template_set"] = True
            meta += [{"source": "template", "kind": "distance", "level": "1/1", "position": "anywhere"}
                     for _ in self.template_owners]
        C = self.candidates(X)
        assert C.shape[1] == len(meta), (C.shape, len(meta))
        self.keep, self.selection_log = self.select(C.numpy(), Y, meta)
        self.kept_names = [(meta[k]["source"], meta[k]["kind"]) for k in self.keep]
        self.n_candidates = C.shape[1]
        self.view_width = {name: sum(m["source"] == name for m in meta) for name, Z in views}
        self.kept_views = {meta[k]["source"] for k in self.keep}
        if self.catch22:                                 # for new data: only the kept catch22 features, per view
            self.c22_kept, start = {}, len(meta) - 44 - (len(self.template_owners) if self.templates else 0)
            for v, name in enumerate(["series", "change (smoothed)"]):
                columns = [k - start - 22 * v for k in self.keep if start + 22 * v <= k < start + 22 * (v + 1)]
                features = [c22_names.feature_names[c] for c in columns]
                self.c22_kept[name] = (columns, Catch22(features = features, replace_nans = True).fit(
                    self.c22_input(X[:2], "series"), Y[:2]) if features else None)
        return C[:, self.keep]

    def select(self, C, Y, meta):
        """Relevance check + greedy selection; returns (chosen columns, what happened to every candidate)."""
        scores, p_values = anova(C, Y)
        relevant = p_values < 0.05
        if relevant.sum() < 10:                          # almost nothing separates the classes: rank all instead
            relevant = scores > 0
        weak = np.zeros(len(scores), dtype = bool)
        if self.fill:                                    # fill the budget: go on down the ranking past p >= 0.05
            weak = ~relevant & (scores > 0)              # (p and F give the same order, so these come last)
            relevant = relevant | weak
        order = np.argsort(-scores)                      # rank = "anova": down the ANOVA ranking
        cut = np.zeros(len(scores), dtype = bool)
        eligible = np.flatnonzero(relevant)
        if self.rank == "both" and len(eligible) > self.budget:
            # as v4 (run.py --tabpfn_rank both): ANOVA keeps the budget + half of the rest, then ExtraTrees ranks them
            # by importance (it also sees features that matter only together with others)
            from sklearn.ensemble import ExtraTreesClassifier
            pool = eligible[np.argsort(-scores[eligible])[:self.budget + (len(eligible) - self.budget) // 2]]
            cut[eligible] = True
            cut[pool] = False
            relevant = relevant & ~cut
            trees = ExtraTreesClassifier(n_estimators = 200, max_features = 0.1, criterion = "entropy", n_jobs = -1,
                                         random_state = self.seed).fit(C[:, pool], Y)
            order = pool[np.argsort(-trees.feature_importances_)]
        spread = C.std(0)
        S = (C - C.mean(0)) / np.where(spread > 0, spread, 1)
        n = len(C)
        taken = np.zeros((n, self.budget), dtype = np.float32)
        kept, outcome, copy_of = [], ["irrelevant"] * len(meta), [""] * len(meta)
        for j in np.flatnonzero(cut):
            outcome[j] = "below the ANOVA cut"
        for j in order:
            if not relevant[j]:
                continue
            if len(kept) == self.budget:
                outcome[j] = "over budget"
                continue
            if kept:
                r = np.abs(taken[:, :len(kept)].T @ S[:, j]) / n
                if r.max() >= 0.95:
                    outcome[j], copy_of[j] = "near-copy", meta[kept[int(r.argmax())]]["source"]
                    continue
            taken[:, len(kept)] = S[:, j]
            kept.append(j)
            outcome[j] = "kept (weak)" if weak[j] else "kept"
        log = [dict(m, outcome = o, copy_of = c) for m, o, c in zip(meta, outcome, copy_of)]
        return np.sort(np.array(kept, dtype = int)), log

    def transform(self, X):
        return self.candidates(X, kept_only = True)[:, self.keep]
