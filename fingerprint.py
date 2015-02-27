__author__ = 'Jan'


# fingerprint song
import numpy as np


class Table():
    """"Table of fingerprints
    Contains a matrix of fingerprints (as rows) and a list of the corresponding songids.
    Allows for more than one fingerprint per song.

    :param c: corpus
    :param fptype: fingerprint function
        choose from: mixedfp, pitchfp, chromafp
    :param params: dictionary of parameters to the fingerprint function
        e.g. {'win':4, 'diagfactor':0, 'norm':True}
    """

    def __init__(self, c, fptype, params):
        self.c = c
        self.ids, self.fps, self.fts = self.compute_fps(fptype, params)

    def compute_fps(self, fptype, params):
        # NOTE:
        # Using vstack on the fly instead of appending to a list is not feasible without checking for the dimensionality
        #   of the fingerprint. Also, using a list and calling vstack just once is actually quite efficient.
        fps, ids, fts = [], [], []
        for i, s in enumerate(self.c.collection):
            songfps, songids, songfeatures = fptype(s, **params)
            ids = ids + [songids]
            fps = fps + [songfps]
            fts = fts + [songfeatures]
        return np.hstack(ids), np.vstack(fps), fts

    def recompute_fps(self, weights=(1,), norm='whiten', key_handling='trans'):
        fps, ids = [], []
        for i, s in enumerate(self.c.collection):
            features = self.fts[i]
            songfps, songids = fp_from_features(s.songid, features, weights, norm, key_handling, verbose=False)
            ids = ids + [songids]
            fps = fps + [songfps]
        self.ids = np.hstack(ids)
        self.fps = np.vstack(fps)


# fingerprinting functions


def pbhfp(s, win=0.5, diagfactor=0, normfunction='whiten', keyhandling='trans'):
    features = [s.get_pitchbihist(win=win, aligned=True, diagfactor=diagfactor)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def cccfp(s, diagfactor=0.5, mode='corr', normfunction='whiten', keyhandling='trans'):
    features = [s.get_chromacorr(diagfactor=diagfactor, mode=mode)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def cbhfp(s, win=0.5, diagfactor=0.5, normfunction='whiten', keyhandling='trans'):
    features = [s.get_chromabihist(win=win, diagfactor=diagfactor)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def harmfp(s, diagfactor=1, normfunction='whiten', keyhandling='trans'):
    features = [s.get_harmonisation(diagfactor=diagfactor)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def mixedfp(s, weights=(1, 1, 1), normfunction='whiten', keyhandling='trans'):
    m0 = s.get_pitchbihist(diagfactor=0)
    m1 = s.get_chromacorr(diagfactor=0.5)
    m2 = s.get_harmonisation(diagfactor=1)
    features = [m0, m1, m2]
    fps, ids = fp_from_features(s.songid, features, weights=weights, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def pitchfp(s, win=0.5, diagfactor=0, normfunction='sum', keyhandling='trans'):
    features = [s.get_pitchtrihist(win, diagfactor, norm=False)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def chromafp(s, win=0.5, diagfactor=0, normfunction='whiten', keyhandling='trans'):
    features = [s.get_chromatrihist(win, diagfactor, norm=False)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


def xgramfp(s, win=0.5, diagfactor=0, mode='min', normfunction='sum', keyhandling='trans'):
    features = [s.get_xgram(win, diagfactor, mode=mode)]
    fps, ids = fp_from_features(s.songid, features, normfunction=normfunction, keyhandling=keyhandling)
    return fps, ids, features


# support functions


def fp_from_features(songid, features, weights=(1,), normfunction='whiten', keyhandling='trans', verbose=True):
    if verbose:
        print 'Fingerprinting song {}'.format(songid)
    flatfeatures = [flatten_feature(feat, keyhandling) for feat in features]
    fps = concat_features(flatfeatures, weights=weights, normfunction=normfunction)
    ids = np.tile(songid, fps.shape[0])
    return fps, ids


def concat_features(x, weights=(1,), normfunction='none'):
    import numpy as np
    if len(weights) == 1 < len(x):
        weights = (weights,) * len(x)
    for i in range(len(x)):
        x[i] = weights[i] * normalize_fp(x[i], normfunction)
    return np.hstack(x)


def flatten_feature(x, keyhandling='trans', verbose=False):
    """ Performs optional transposition + flattening
    Returns a vertical array of flat (transpositions of / transposition-invariant slices of) x
    """

    if keyhandling[:5] == 'trans':
        xflat = all_keys(x)
    elif keyhandling[:8] == 'keyinvar':
        xflat = key_invariant_slices(x)
    elif keyhandling[:8] == 'diff':
        xflat = [diffgram(x)]
    elif keyhandling[:4] == 'none':
        xflat = [x.flatten()]
    else:
        raise NameError('keyhandling instruction unclear')
    if verbose:
        print 'keyhandling = ' + keyhandling
        print 'flattened features:'
        print [row.shape for row in xflat]
    return np.vstack(xflat)


# key handling


def all_keys(x):
    return [transpose_fp(x, trans).flatten() for trans in range(12)]


def transpose_fp(x, t):
    import numpy as np
    for ax in range(len(x.shape)):
        x = np.roll(x, t, ax)
    return x


def diffgram(x):
    return np.sum(np.vstack(key_invariant_slices(x)), axis=0)


def key_invariant_slices(x):
    if not np.mod(x.shape[0], 12) == 0:
        print 'Warning: probably not appropriate to use keyinvariant slicing for key handling'
    rng = np.arange(x.shape[0])
    if len(x.shape) == 3:
        return [np.array([[x[i, j-i, k-i] for k in rng] for j in rng]).flatten() for i in rng]
    elif len(x.shape) == 2:
        return [np.array(x[i, j - i] for j in rng) for i in rng]


# normalization functions


def threshold_fp(x, par, mode='topk', verbose=False):

    if mode == 'topk':
        k = par
        thr = np.sort(x)[-k]
    elif mode == 'topr':  # r for relative
        r = par
        n = len(x)
        k = np.round(r * n)
        thr = np.sort(x)[-k]
    elif mode == 'tops':  # s for share or sum
        s = par
        fpsum = np.sum(x)
        fpsort = np.sort(x)
        thr = fpsort[np.argmax(np.cumsum(fpsort / fpsum) > (1 - s))]
    return float(x > thr)


def normalize_fp(x, normfunction='none'):
    for row in range(len(x)):
        x[row] = normalize_array(x[row], normfunction=normfunction)
    return x


def normalize_array(x, normfunction='none'):
    """
    Normalize an array.
    :param x: array
    :param normfunction: normalization mode,
        'max' divides by maximum
        'sum' divides by sum (beware: scales with length of x)
        'mean divides by mean
        'std' divides by the standard deviation
        'whiten' substracts the mean before dividing by the standard deviation
        'vector' divides by 2-norm (beware: scales with length of x)
    :return: normalized array, same shape as x
    """
    # TODO: now, 'whiten' is default around most of this module. that should probably be changed to 'vector'
    # main advantage of 'vector': more logical in context where cosine distance applies
    # specifically, it works well with both positive definite data and data centered around zero since
    #     - sum/mean is bad for zero-centered data
    #     - whiten will not preserve zeros in either kind of data
    # downside: probably more sensitive to outliers than 'sum' or 'mean'
    eps = 0.0001
    if normfunction == 'max':
        x = x * 1.0 / max(x)
        assert (max(x)-1 < eps)
    if normfunction == 'sum':
        x = x * 1.0 / sum(x)
        assert (sum(x)-1 < eps)
    elif normfunction == 'mean':
        x = x * 1.0 / np.mean(x)
        assert (np.mean(x)-1 < eps)
    elif normfunction == 'std':
        x = x * 1.0 / np.std(x)
    elif normfunction == 'whiten':
        x = x - np.mean(x)
        x = x * 1.0 / np.std(x, ddof=1)
    elif normfunction == 'vector':
        xnorm = np.sqrt(np.sum(np.power(x,2)))
        x = x * 1.0 / xnorm
    elif normfunction == 'none':
        pass
    else:
        print 'Normalization function unknown, please try another one (eg. sum, whiten or none).'
    return x

# # generic fingerprinting function
# #   takes a song + tuple of *bounded* feature functions cfr. mysong.get_chromacorr, etc.
#
#
# def genfp(s, fpfunctions, params={}, weights=1, norm='whiten', transpose=True):
#     import numpy as np
#
#     print 'Fingerprinting song {}'.format(s.songid)
#     mats = []
#     for fpfunc in fpfunctions:
#         mats = mats + flatten_feature(fpfunc(**params), transpose=transpose)
#     fps = concat_features(mats, weights=weights, normfunction=norm)
#     ids = np.tile(s.songid, fps.shape[0])
#     return ids, fps


# keyinvariant transform
#   to be reimplemented using 1d/2d dct/fft or similar.