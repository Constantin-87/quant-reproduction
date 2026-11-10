# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# QUANT: A Minimalist Interval Method for Time Series Classification
# https://arxiv.org/abs/2308.00928
# (update to handle multivariate per https://github.com/angus924/aaltd2024)

# quant_tabpfn.py: a copy of the authors' quant.py, changed for TabPFN with a small feature budget.
# The authors' lines are kept, commented out with "# ORIGINAL:", next to each change:
#   1. fewer levels: depth 5 instead of 6 (the smallest pieces, mostly noise, are dropped)
#   2. a fixed number of quantiles per piece instead of one per 4 points: with depth 4 and lags 4, 16 about 450 features
#   3. no shifted pieces (the half-overlapping copies of every level)
#   4. no subtraction of the piece's average (a trick for trees, which look at one feature at a time)
#   5. per piece also its mean, spread and trend (more information per feature than more quantiles)
#   6. SupervisedExtractor: candidates from several views (+ optionally catch22 and pattern templates),
#      a relevance check and a greedy selection of at most `budget` non-redundant features, all decided on the training labels
#   7. optional (fraction > 0): a fraction of the authors' quantiles (one per 4 points) instead of num_quantiles, odd
#      so that the median stays in, e.g. 0.2 = 25 instead of 125 for a piece of 500 points

import numpy as np

import torch, torch.nn.functional as F

# == generate intervals ========================================================

# the pieces (start, end) of every level: the whole series, its halves, quarters, ...
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

def quantile_count(n, num_quantiles = 2, fraction = 0):
    """The number of quantiles of a piece of n > 1 points (change 2 and 7; the authors: 1 + (n - 1) // 4)."""
    if fraction > 0:                                    # change 7: a fraction of the authors' count, odd (median in)
        k = max(1, round((1 + (n - 1) // 4) * fraction))
        return min(k + 1 - k % 2, n)
    return min(num_quantiles, n)

# the quantiles of every piece
# change 2: a fixed number of quantiles per piece instead of one per div = 4 points
# ORIGINAL: def f_quantile(X, div = 4):
def f_quantile(X, num_quantiles = 2, fraction = 0):

    n = X.shape[-1]

    if n == 1:                                          # one point: the value itself

        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])

    else:

        # change 2 and 7: a fixed number or a fraction of the quantiles per piece
        # ORIGINAL: num_quantiles = 1 + (n - 1) // div
        num_quantiles = quantile_count(n, num_quantiles, fraction)

        if num_quantiles == 1:                          # one quantile: the median

            quantiles = X.quantile(torch.tensor([0.5]), dim = -1).permute(1, 2, 0)

            return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

        else:                                           # evenly spaced, from the minimum to the maximum

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

# the features of one view: per piece its quantiles + mean, spread and trend
class IntervalModel():

    # change 1 and 2: depth 5 instead of 6, a fixed number of quantiles per piece instead of one per div = 4 points
    # ORIGINAL: def __init__(self, input_length, depth = 6, div = 4):
    def __init__(self, input_length, depth = 5, num_quantiles = 2, fraction = 0):

        assert num_quantiles >= 1
        assert depth >= 1

        self.num_quantiles = num_quantiles                  # change 2
        self.fraction = fraction                            # change 7

        self.intervals = \
        make_intervals(
            input_length = input_length,
            depth        = depth,
        )

    def fit(self, X, Y):                                # nothing to learn

        pass

    def transform(self, X):

        features = []

        for a, b in self.intervals:                     # every piece

            features.append(
                f_quantile(X[..., a:b], num_quantiles = self.num_quantiles, fraction = self.fraction).squeeze(1)
            )
            features.append(f_summary(X[..., a:b]))     # change 5

        return torch.cat(features, -1)

    def fit_transform(self, X, Y):

        self.fit(X, Y)

        return self.transform(X)

# == quant =====================================================================

# v3: the interval features of QUANT's four views
class Quant():

    # change 1 and 2: depth 5 instead of 6, a fixed number of quantiles per piece instead of one per div = 4 points
    # ORIGINAL: def __init__(self, depth = 6, div = 4):
    def __init__(self, depth = 5, num_quantiles = 2):

        assert depth >= 1
        assert num_quantiles >= 1

        self.depth = depth
        self.num_quantiles = num_quantiles                  # change 2

        # the four views: the series, its smoothed change, change of change, rhythm
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
                num_quantiles = self.num_quantiles,     # change 2
            )

            features.append(
                self.models[index].fit_transform(Z, Y)
            )

        self.fitted = True

        return torch.cat(features, -1)

# == change 6: a supervised extractor ============================================
#
# Instead of returning every feature and leaving the choice to a later filter, it looks at the training labels and
# returns at most `budget` strong, non-redundant features itself:
#   1. candidates: the interval features of several views of the series (QUANT's four, lag-d differences),
#      optionally the catch22 features (whole-series behaviour) and pattern templates (short windows
#      from training series);
#   2. a relevance check: candidates that do not separate the classes on the training data (ANOVA, p >= 0.05) go;
#   3. greedy selection: down the ANOVA ranking (or the ExtraTrees importance, rank = "both"), a feature is taken
#      only if it is not a near-copy (|correlation| >= 0.95) of one already taken, up to the budget.
# Everything is decided in fit_transform on the training data; transform computes the same candidates for new data
# (only the views and catch22 features with kept features) and returns the chosen columns. selection_log records
# what happened to every candidate (notebook 04).
# fill = True makes the limits relative to the dataset: the selection goes on down the ranking past p >= 0.05
# ("kept (weak)") until the budget is full, and when even that is not enough (few candidates, many near-copies) the
# pieces are made finer (depth + 1, up to max_depth) and the selection is redone; catch22 and the templates do not
# depend on the piece size, so they are computed once and shared by these rounds.

def anova(features, Y):
    """ANOVA F-scores and p-values of every column (constant columns: F = 0, p = 1)."""
    import warnings
    from sklearn.feature_selection import f_classif
    with warnings.catch_warnings(), np.errstate(all = "ignore"):     # no warnings for constant columns
        warnings.simplefilter("ignore")
        scores, p_values = f_classif(features, Y)
    scores = np.nan_to_num(scores, nan = 0.0)           # NaN -> 0
    p_values = np.nan_to_num(p_values, nan = 1.0)       # NaN -> 1
    return scores, p_values

def template_distances(S, T, batch = 256):
    """Smallest mean squared distance between each template (rows of T, length w) and any window of each series
    (rows of S): (cases, templates). Uses (x - t)^2 = x^2 - 2 x t + t^2, so no window is copied."""
    w = T.shape[-1]
    out = []
    for start in range(0, S.shape[0], batch):           # in batches of cases, to bound the memory
        x = S[start:start + batch].unsqueeze(1)          # (cases, 1, length)
        cross = F.conv1d(x, T.unsqueeze(1))              # (cases, templates, windows): the sum of x * t per window
        x2 = F.avg_pool1d(x ** 2, w, 1) * w              # (cases, 1, windows): the sum of x^2 per window
        t2 = (T ** 2).sum(-1)[None, :, None]             # (1, templates, 1): the sum of t^2
        d = (x2 - 2 * cross + t2).clamp(min = 0) / w     # mean squared distance per window
        out.append(d.min(-1).values)                     # the closest window
    return torch.cat(out)

class SupervisedExtractor():

    QUANT_VIEWS = ("series", "change (smoothed)", "change of change", "rhythm (|FFT|)")   # get quantile_fraction
    C22_VIEWS = ("series", "change (smoothed)")         # catch22 is computed on these two

    def __init__(self, depth = 4, num_quantiles = 2, lags = False, catch22 = False, templates = False, budget = 250,
                 fill = False, max_depth = 6, c22_shorten = False, rank = "anova", quantile_fraction = 0, seed = 0):
        self.depth = depth                               # depth of the first round
        self.num_quantiles = num_quantiles               # quantiles per piece (the lag views)
        self.quantile_fraction = quantile_fraction       # change 7 (v4): QUANT's views, 0 = num_quantiles there too
        self.lags = lags                                 # True: add the lag-d views
        self.catch22 = catch22                           # True: add the catch22 features
        self.templates = templates                       # True: add the template distances
        self.budget = budget                             # at most this many features kept
        self.fill = fill                                 # True: fill the budget (weak features, finer pieces)
        self.max_depth = max_depth                       # the finest depth when filling
        self.c22_shorten = c22_shorten                   # True: catch22 on a shortened copy (c22_factor)
        self.rank = rank                                 # "anova", or "both" = ANOVA then ExtraTrees
        self.seed = seed                                 # for the templates and ExtraTrees
        self.round_cache = {}                            # training only: blocks shared by the fill rounds

    def lag_list(self, length):
        """The lag-d views (lags = True): d = 2, 4, 8, ... up to a quarter of the length (at most 64)."""
        lags = []
        if not self.lags:
            return lags
        for k in range(1, 7):                            # d = 2 ... 64
            d = 2 ** k
            if d <= length // 4:
                lags.append(d)
        return lags

    def views(self, X):
        """Every view the interval features are computed on: {name: series}, in a fixed order."""
        views = {}
        views["series"] = X
        views["change (smoothed)"] = F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1)
        views["change of change"] = X.diff(n = 2)
        views["rhythm (|FFT|)"] = torch.fft.rfft(X).abs()
        for d in self.lag_list(X.shape[-1]):
            views[f"lag-{d}"] = X[..., d:] - X[..., :-d]      # x(t) - x(t - d)
        return views

    def interval_meta(self, name, model, length):
        """source, kind, level (piece size) and position of every interval feature, in the column order."""
        meta = []
        for a, b in model.intervals:
            a = int(a)
            b = int(b)
            n = b - a                                    # points in the piece
            level = int(round(np.log2(length / n)))      # 0 = whole series, 1 = halves, 2 = quarters, ...
            centre = (a + b) / 2 / length                # the piece's centre, from 0 to 1
            if level == 0:
                position = "whole"
            elif centre < 1 / 3:
                position = "early"
            elif centre > 2 / 3:
                position = "late"
            else:
                position = "middle"
            n_quantiles = quantile_count(n, model.num_quantiles, model.fraction)
            kinds = ["quantile"] * n_quantiles + ["mean", "spread", "trend"]     # as IntervalModel.transform
            for kind in kinds:
                meta.append({"source": name, "kind": kind, "level": f"1/{2 ** level}", "position": position})
        return meta

    def fit_templates(self, S, Y):
        """Pattern templates: windows of training series, the best ones per class (one class against the rest)."""
        length = S.shape[1]
        w = max(3, min(64, length // 8))                 # window length
        rng = np.random.default_rng(self.seed)
        classes = np.unique(Y)
        per_class = max(1, min(3, 60 // len(classes)))  # kept per class: 2 classes 3 each, 60 classes 1 each
        tries = max(per_class, min(20, 200 // len(classes)))   # random windows tried per class
        templates = []
        owners = []                                      # the class of each template
        for c in classes:
            rows = np.flatnonzero(Y == c)                # the training cases of class c
            picks = rng.choice(rows, size = tries)       # random cases
            starts = rng.integers(0, length - w + 1, size = tries)   # random window starts
            windows = []
            for r, s in zip(picks, starts):
                windows.append(S[r, s:s + w])
            candidates = torch.stack(windows)
            distances = template_distances(S, candidates).numpy()
            scores, _ = anova(distances, (Y == c).astype(int))      # class c against the rest
            best = np.argsort(-scores)[:per_class]
            for k in best:
                templates.append(candidates[k])
                owners.append(c)
        self.template_set = torch.stack(templates)
        self.template_owners = owners

    def c22_factor(self, Z, energy = 0.99, min_length = 32, keep_below = 256):
        """c22_shorten: by how much this view can be shortened for catch22, from the training series themselves. Series
        shorter than keep_below points stay as they are (catch22 is fast on them). For longer ones, the mean spectrum
        (|FFT|^2) says how the detail is spread over fast and slow changes; the factor k is the largest that keeps the
        changes holding `energy` of it (the copy keeps changes up to 1/k of the fastest possible) and leaves at least
        min_length points. Smooth series get a large k, noisy or spiky ones k = 1 (not shortened)."""
        if not self.c22_shorten or Z.shape[-1] < keep_below:
            return 1
        centred = Z[:, 0, :] - Z[:, 0, :].mean(-1, keepdim = True)
        power = (torch.fft.rfft(centred).abs() ** 2).mean(0)[1:]   # mean spectrum, without the constant
        if power.sum() <= 0:                             # a constant series
            return 1
        needed = int((power.cumsum(0) / power.sum() < energy).sum()) + 1          # frequencies holding `energy`
        return int(max(1, min(len(power) // needed, Z.shape[-1] // min_length)))

    def c22_input(self, Z, name):
        """The series catch22 is computed on: Z, or with c22_shorten a copy with the mean of every k points."""
        k = self.c22_k[name]
        if k > 1:
            Z = F.avg_pool1d(Z, k)
        return Z.double().numpy()

    def candidates(self, X, kept_only = False):
        """All candidate features of X (cases, candidates), in a fixed column order. kept_only = True (new data, after
        fit_transform): only the views and catch22 features with kept features are computed; the others stay 0 (they
        are not returned)."""
        views = self.views(X)
        blocks = []
        # 1. the interval features of every view
        for name, Z in views.items():
            if kept_only and name not in self.kept_views:
                blocks.append(torch.zeros(len(X), self.view_width[name]))     # not needed: zeros
            else:
                blocks.append(self.models[name].transform(Z))
        # 2. catch22 (22 features per view)
        if self.catch22:
            for name in self.C22_VIEWS:
                Z = views[name]
                if kept_only:                            # new data: only the kept columns
                    block = torch.zeros(len(X), 22)
                    columns, transformer = self.c22_kept[name]
                    if columns:
                        values = transformer.transform(self.c22_input(Z, name))
                        block[:, columns] = torch.tensor(values, dtype = torch.float32)
                    blocks.append(block)
                else:                                    # training: computed once, reused by the fill rounds
                    if name not in self.round_cache:
                        values = self.c22[name].transform(self.c22_input(Z, name))
                        self.round_cache[name] = torch.tensor(values, dtype = torch.float32)
                    blocks.append(self.round_cache[name])
        # 3. the template distances
        if self.templates:
            if kept_only:
                blocks.append(template_distances(X[:, 0, :], self.template_set))
            else:                                        # training: computed once, reused by the fill rounds
                if "templates" not in self.round_cache:
                    self.round_cache["templates"] = template_distances(X[:, 0, :], self.template_set)
                blocks.append(self.round_cache["templates"])
        C = torch.cat(blocks, -1)
        return torch.nan_to_num(C, nan = 0.0, posinf = 0.0, neginf = 0.0)     # NaN and inf -> 0

    def fit_transform(self, X, Y):
        """Choose the features on the training data, and return them."""
        self.round_cache = {}
        depth = self.depth
        while True:
            C = self.fit_round(X, Y, depth)
            if not self.fill or len(self.keep) >= self.budget or depth >= self.max_depth:
                break                                    # no filling, budget full, or the finest pieces reached
            depth += 1                                   # fill: finer pieces, then select again
        self.round_cache = {}                            # emptied: new data (transform) is always computed
        return C

    def fit_round(self, X, Y, depth):
        """Candidates with pieces down to 1/2**(depth - 1), and the selection on them."""
        from aeon.transformations.collection.feature_based import Catch22
        from aeon.transformations.collection.feature_based._catch22 import feature_names as C22_NAMES

        # 1. one interval model per view, and meta = a description of every candidate column
        views = self.views(X)
        self.models = {}
        meta = []
        for name, Z in views.items():
            if name in self.QUANT_VIEWS:
                fraction = self.quantile_fraction
            else:
                fraction = 0
            self.models[name] = IntervalModel(Z.shape[-1], depth = depth, num_quantiles = self.num_quantiles,
                                              fraction = fraction)
            meta += self.interval_meta(name, self.models[name], Z.shape[-1])
        # 2. catch22
        if self.catch22:
            self.c22 = {}                                # the fitted catch22 of each view
            self.c22_k = {}                              # the shortening factor of each view
            for name in self.C22_VIEWS:
                Z = views[name]
                self.c22_k[name] = self.c22_factor(Z)
                self.c22[name] = Catch22(replace_nans = True).fit(self.c22_input(Z, name), Y)
                for feature in C22_NAMES:
                    meta.append({"source": "catch22", "kind": f"{feature} ({name})", "level": "1/1",
                                 "position": "whole"})
        # 3. templates (chosen in the first round only)
        if self.templates:
            if "template_set" not in self.round_cache:
                self.fit_templates(X[:, 0, :], Y)
                self.round_cache["template_set"] = True
            for _ in self.template_owners:
                meta.append({"source": "template", "kind": "distance", "level": "1/1", "position": "anywhere"})
        # 4. all candidates, and the selection
        C = self.candidates(X)
        assert C.shape[1] == len(meta), (C.shape, len(meta))
        self.keep, self.selection_log = self.select(C.numpy(), Y, meta)
        self.n_candidates = C.shape[1]
        # 5. for new data: the width of each view, and the views with kept features
        self.view_width = {}
        for name in views:
            width = 0
            for m in meta:
                if m["source"] == name:
                    width += 1
            self.view_width[name] = width
        self.kept_views = set()
        for k in self.keep:
            self.kept_views.add(meta[k]["source"])
        # 6. for new data: only the kept catch22 features, per view
        if self.catch22:
            n_templates = len(self.template_owners) if self.templates else 0
            first_c22 = len(meta) - 44 - n_templates     # the column of the first catch22 feature
            self.c22_kept = {}
            for v, name in enumerate(self.C22_VIEWS):
                first = first_c22 + 22 * v               # this view's 22 columns start here
                columns = []                             # the kept ones, 0 to 21
                for k in self.keep:
                    if first <= k < first + 22:
                        columns.append(k - first)
                features = [C22_NAMES[c] for c in columns]
                transformer = None
                if features:                             # aeon needs a fit, it learns nothing: 2 cases are enough
                    transformer = Catch22(features = features, replace_nans = True).fit(
                        self.c22_input(X[:2], "series"), Y[:2])
                self.c22_kept[name] = (columns, transformer)
        return C[:, self.keep]

    def select(self, C, Y, meta):
        """Relevance check + greedy selection; returns (chosen columns, what happened to every candidate)."""
        # 1. relevance: the columns that separate the classes (ANOVA p < 0.05)
        scores, p_values = anova(C, Y)
        relevant = p_values < 0.05
        if relevant.sum() < 10:                          # almost nothing separates the classes: rank all instead
            relevant = scores > 0
        weak = np.zeros(len(scores), dtype = bool)       # p >= 0.05, used only to fill the budget
        if self.fill:
            weak = ~relevant & (scores > 0)              # (p and F give the same order, so these come last)
            relevant = relevant | weak
        # 2. the order to try them in
        order = np.argsort(-scores)                      # rank = "anova": the highest F first
        cut = np.zeros(len(scores), dtype = bool)        # dropped before the ExtraTrees ranking
        eligible = np.flatnonzero(relevant)              # the column numbers still in
        if self.rank == "both" and len(eligible) > self.budget:
            # as v3 (run.py --tabpfn_rank both): ANOVA keeps the budget + half of the rest, then ExtraTrees ranks them
            # by importance (it also sees features that matter only together with others)
            from sklearn.ensemble import ExtraTreesClassifier
            pool_size = self.budget + (len(eligible) - self.budget) // 2
            pool = eligible[np.argsort(-scores[eligible])[:pool_size]]     # the best by ANOVA
            cut[eligible] = True
            cut[pool] = False
            relevant = relevant & ~cut
            trees = ExtraTreesClassifier(n_estimators = 200, max_features = 0.1, criterion = "entropy", n_jobs = -1,
                                         random_state = self.seed).fit(C[:, pool], Y)
            order = pool[np.argsort(-trees.feature_importances_)]        # the most important first
        # 3. greedy: take them in order, skip near-copies of a kept one (|correlation| >= 0.95)
        spread = C.std(0)
        S = (C - C.mean(0)) / np.where(spread > 0, spread, 1)            # standardised columns
        n = len(C)
        taken = np.zeros((n, self.budget), dtype = np.float32)          # the standardised kept columns
        kept = []
        outcome = ["irrelevant"] * len(meta)
        copy_of = [""] * len(meta)
        for j in np.flatnonzero(cut):
            outcome[j] = "below the ANOVA cut"
        for j in order:
            if not relevant[j]:
                continue
            if len(kept) == self.budget:
                outcome[j] = "over budget"
                continue
            if len(kept) > 0:
                r = np.abs(taken[:, :len(kept)].T @ S[:, j]) / n         # correlation with every kept column
                if r.max() >= 0.95:
                    outcome[j] = "near-copy"
                    copy_of[j] = meta[kept[int(r.argmax())]]["source"]
                    continue
            taken[:, len(kept)] = S[:, j]
            kept.append(j)
            if weak[j]:
                outcome[j] = "kept (weak)"
            else:
                outcome[j] = "kept"
        # 4. the log: every candidate's description + what happened to it
        log = []
        for j in range(len(meta)):
            entry = dict(meta[j])
            entry["outcome"] = outcome[j]
            entry["copy_of"] = copy_of[j]
            log.append(entry)
        return np.sort(np.array(kept, dtype = int)), log

    def transform(self, X):
        """The chosen features of new data."""
        C = self.candidates(X, kept_only = True)
        return C[:, self.keep]
