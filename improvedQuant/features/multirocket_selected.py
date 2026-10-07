# Compute only some of MultiRocket's features ("train with many, predict with few"), for v3-sep.
#
# In v3-joint, MultiRocket computes all its features (49,728 with 6,250 kernels) for the training AND the test set,
# and the ANOVA filter then keeps only a few thousand of them. The training set needs all of them once, to score
# them, but the test set only needs the kept ones. aeon's MultiRocket always computes everything, so this module
# computes just the chosen columns, with the parameters MultiRocket fitted on the training data. The values are
# the same as aeon's (a copy of aeon's calculation, restricted to the chosen columns; checked against
# MultiRocket.transform(X)[:, columns]).
#
# How aeon's MultiRocket (aeon 1.6, one channel) lays out its features:
#   - two series: the time series itself ("base") and its first difference ("diff")
#   - for each series: n_dilations x 84 kernels x a few biases = n_f "kernel features" (6,216 for 6,250
#     kernels), numbered dilation by dilation, then kernel by kernel, then bias by bias
#   - each kernel feature gives 4 pooled values: PPV (proportion of positive values), LSPV (longest stretch of
#     positive values), MPV (mean of positive values) and MIPV (mean index of positive values)
#   - columns: base PPV [0, n_f), base LSPV [n_f, 2 n_f), base MPV, base MIPV, then the same 4 blocks for diff
# So each column number tells which (series, dilation, kernel, bias, pooling) it is, and computing only the
# chosen columns means: only the needed (dilation, kernel) convolutions, and for each only the needed biases.
#
# Only for univariate series and MultiRocket(normalise = False), as used in run.py.

import numpy as np
from numba import njit, prange, set_num_threads

N_KERNELS = 84      # MultiRocket's (and MiniRocket's) fixed kernels


def _plan(multirocket, columns):
    """Work out which convolutions and biases the chosen columns need, and where each result goes.

    Returns
      conv:       (n_conv, 3) array, one row per needed convolution: series (0 base, 1 diff), dilation index,
                  kernel index; sorted, so all kernels of one dilation come one after the other
      item_start: where each convolution's biases start in item_bias / item_out (n_conv + 1 entries)
      item_bias:  the bias of each needed kernel feature
      item_out:   (n_items, 4): output column of its PPV, LSPV, MPV and MIPV, or -1 if that one is not needed
    """
    parameters = [multirocket.parameter, multirocket.parameter1]   # (dilations, features per dilation, biases)
    n_f = N_KERNELS * int(np.sum(parameters[0][1]))
    assert n_f == N_KERNELS * int(np.sum(parameters[1][1])), "base and diff must have the same number of features"

    # first kernel feature of each dilation, per series
    starts = [N_KERNELS * np.concatenate([[0], np.cumsum(p[1])]) for p in parameters]

    wanted = {}     # (series, dilation, kernel) -> {kernel feature: [output column of each pooling]}
    for out, column in enumerate(np.asarray(columns, dtype = np.int64)):
        series, rest = divmod(int(column), 4 * n_f)
        pooling, feature = divmod(rest, n_f)
        dilation = int(np.searchsorted(starts[series], feature, side = "right")) - 1
        kernel = int((feature - starts[series][dilation]) // parameters[series][1][dilation])
        wanted.setdefault((series, dilation, kernel), {}).setdefault(feature, [-1, -1, -1, -1])[pooling] = out

    conv, item_start, item_bias, item_out = [], [0], [], []
    for key in sorted(wanted):
        conv.append(key)
        for feature, outs in sorted(wanted[key].items()):
            item_bias.append(parameters[key[0]][2][feature])
            item_out.append(outs)
        item_start.append(len(item_bias))
    return (np.array(conv, dtype = np.int64).reshape(-1, 3), np.array(item_start, dtype = np.int64),
            np.array(item_bias, dtype = np.float32), np.array(item_out, dtype = np.int64).reshape(-1, 4))


@njit(fastmath = True, parallel = True, cache = True)
def _transform_selected(X, X1, dilations, dilations1, conv, item_start, item_bias, item_out, indices, n_out):
    """aeon's _transform_uni, restricted to the planned convolutions and biases (same arithmetic)."""
    n_cases, n_timepoints = X.shape
    features = np.zeros((n_cases, n_out), dtype = np.float32)

    for example_index in prange(n_cases):
        previous_series, previous_dilation = -1, -1
        C_alpha = np.zeros(1, dtype = np.float32)
        C_gamma = np.zeros((9, 1), dtype = np.float32)
        padding = 0

        for c in range(conv.shape[0]):
            series, dilation_index, kernel_index = conv[c, 0], conv[c, 1], conv[c, 2]

            # the 9 shifted copies of the series for this dilation, once per (series, dilation)
            if series != previous_series or dilation_index != previous_dilation:
                if series == 0:
                    _X = X[example_index]
                    dilation = dilations[dilation_index]
                else:
                    _X = X1[example_index]
                    dilation = dilations1[dilation_index]
                A = -_X                 # alpha * X
                G = _X + _X + _X        # gamma * X
                padding = ((9 - 1) * dilation) // 2

                C_alpha = np.zeros(_X.shape[0], dtype = np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, _X.shape[0]), dtype = np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = n_timepoints - padding    # as in aeon: the base series' length, also for the difference
                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation
                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation
                previous_series, previous_dilation = series, dilation_index

            # the convolution with this kernel
            index_0, index_1, index_2 = indices[kernel_index]
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            # as in aeon: every other (dilation, kernel) pair leaves out the padded ends
            lo, hi = 0, C.shape[0]
            if (dilation_index % 2 + kernel_index) % 2 == 1:
                lo, hi = padding, C.shape[0] - padding
            n = hi - lo

            # the 4 pooled values for each needed bias
            for item in range(item_start[c], item_start[c + 1]):
                _bias = item_bias[item]
                ppv = 0
                last_val = 0
                max_stretch = 0.0
                mean_index = 0
                mean = 0
                for j in range(n):
                    value = C[lo + j]
                    if value > _bias:
                        ppv += 1
                        mean_index += j
                        mean += value + _bias
                    elif value < _bias:
                        stretch = j - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch
                        last_val = j
                stretch = n - 1 - last_val
                if stretch > max_stretch:
                    max_stretch = stretch

                if item_out[item, 0] >= 0:
                    features[example_index, item_out[item, 0]] = ppv / n
                if item_out[item, 1] >= 0:
                    features[example_index, item_out[item, 1]] = max_stretch
                if item_out[item, 2] >= 0:
                    features[example_index, item_out[item, 2]] = mean / ppv if ppv > 0 else 0
                if item_out[item, 3] >= 0:
                    features[example_index, item_out[item, 3]] = mean_index / ppv if ppv > 0 else -1

    return features


def transform_selected(multirocket, X, columns, n_jobs = -1):
    """The chosen MultiRocket feature columns for X, i.e. multirocket.transform(X)[:, columns], computing only them.

    multirocket: a fitted aeon MultiRocket (one channel, normalise = False)
    X:           numpy array (n_cases, 1, length)
    columns:     column numbers of MultiRocket's output, e.g. the ones kept by the filter
    """
    from aeon.transformations.collection.convolution_based import MultiRocket

    if n_jobs >= 1:
        set_num_threads(n_jobs)
    X = X.astype(np.float32)
    X = X.reshape(X.shape[0], X.shape[2])
    X1 = np.diff(X, 1)
    conv, item_start, item_bias, item_out = _plan(multirocket, columns)
    features = _transform_selected(X, X1, multirocket.parameter[0], multirocket.parameter1[0], conv, item_start,
                                   item_bias, item_out, MultiRocket._indices, len(columns))
    return np.nan_to_num(features)      # as aeon does
