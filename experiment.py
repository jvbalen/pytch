__author__ = 'Jan'


# TODO:
# - move groundtruth attribute to corpus

import collection
import fingerprint as fp
import numpy as np


class Experiment(object):
    def __init__(self, qcorpus, ccorpus, groundtruth, fptype=fp.mixedfp, params={}):
        self.qcorpus = qcorpus
        self.ccorpus = ccorpus
        self.groundtruth = groundtruth
        self.fptype = fptype
        self.params = params

        self.qdb, self.cdb = self.compute_fps(fptype, params)
        self.fpdm = None
        self.songdm = None
        self.qids = None
        self.cids = None
        self.results = {}

    def compute_fps(self, fptype, params):
        qdb = fp.Table(self.qcorpus, fptype, params)
        if not params.has_key('keyhandling') or params.get('keyhandling')[:5] == 'trans':  # only queries are transposed
            params['keyhandling'] = 'none'
        cdb = fp.Table(self.ccorpus, fptype, params)
        return qdb, cdb

    def recompute_fps(self, weights=1, norm='whiten', key_handling='transpose'):
        self.qdb.recompute_fps(weights=weights, norm=norm, key_handling=key_handling)
        key_handling = 'none' if key_handling[:5] == 'trans' else key_handling  # only queries are transposed
        self.cdb.recompute_fps(weights=weights, norm=norm, key_handling=key_handling)

    def run(self, distmetric='cosine', nmatches=1, evalmetrics=('map', 'p1', 'r5'), verbose=False):
        if verbose:
            print ('computing fingerprint distance matrix')
        self.fpdm = self.fp_distances(distmetric)
        if verbose:
            print ('computing song distance matrix')
        self.songdm, self.qids, self.cids = self.song_distances(nmatches)
        if verbose:
            print ('evaluation...')
        for metric in evalmetrics:
            self.results[metric] = evaluate(self.songdm, self.groundtruth, metric)
        if verbose:
            print 'results:'
            print self.results

    def run_z(self, distmetric='cosine', nmatches=1, evalmetrics=('map', 'p1', 'r5'), verbose=False):
        if verbose:
            print ('computing fingerprint distance matrix')
        self.fpdm = self.fpdistances_z(distmetric)
        # self.fpdm = self.tfidf_dist(distmetric, tfthr=tfidf, verbose=verbose)
        if verbose:
            print ('computing song distance matrix')
        self.songdm, self.qids, self.cids = self.song_distances(nmatches)
        if verbose:
            print ('evaluation...')
        for metric in evalmetrics:
            self.results[metric] = evaluate(self.songdm, self.groundtruth, metric)
        if verbose:
            print 'results:'
            print self.results

    def fp_distances(self, metric='cosine'):
        from scipy import spatial

        return spatial.distance.cdist(self.qdb.fps, self.cdb.fps, metric)

    # def tfidf_dist(self, metric='cosine', tfthr=10, verbose=False):
    #     from scipy import spatial
    #     tf_q = self.qdb.fps
    #     tf_c = self.cdb.fps
    #     idf = -np.log10(1. + np.sum(tf_c > tfthr, axis=0))
    #     if verbose:
    #         print 'mean(tf > thr): ' + str(np.mean(tf_c > tfthr))
    #     return spatial.distance.cdist(tf_q * idf, tf_c * idf, metric)

    def fpdistances_z(self, metric='cosine'):
        from scipy import spatial
        tf_q = self.qdb.fps
        tf_c = self.cdb.fps
        z_q = (tf_q - np.mean(tf_c, axis=0)) / np.std(tf_c, axis=0)
        z_c = (tf_c - np.mean(tf_c, axis=0)) / np.std(tf_c, axis=0)
        return spatial.distance.cdist(z_q, z_c, metric)

    def song_distances(self, nmatches=1, cumul=False):

        qids = np.unique(self.qdb.ids)
        cids = np.unique(self.cdb.ids)
        songdm = np.zeros((len(qids), len(cids)))
        for i, q in enumerate(qids):
            for j, c in enumerate(cids):
                if self.fpdm is None:
                    print 'Error: first, fingerprint distance matrix must be computed, then song distance matrix.'
                d = self.fpdm[np.ix_(self.qdb.ids == q, self.cdb.ids == c)]
                dsorted = np.sort(d.flatten())
                if cumul:
                    dsorted = np.cumsum(dsorted)
                songdm[i, j] = dsorted[nmatches - 1]
        return songdm, qids, cids

    def log(self, filename='autolog.txt', comment=''):
        fun = self.fptype.__name__ + '\n'
        msg = str(self.params) + '\n' + comment + '    results:' + '\n'
        res = '    ' + str(self.results) + '\n'
        with open(filename, mode='a') as f:
            f.write(msg + res)
            f.close()


class TestExperiment(Experiment):
    def __init__(self, qdb, cdb):
        Experiment.__init__(self, collection.Collection(), collection.Collection(), [])
        self.qdb = qdb
        self.cdb = cdb


def evaluate(distances, groundtruth, metric='map'):
    nqueries = len(distances)
    assert nqueries == len(groundtruth)
    # following line: list with for every query an array with the ranks of the relevant documents
    ranks = [1 + rank(distances[q])[groundtruth[q] == 1] for q in range(nqueries)]
    if metric == 'map':
        aps = [ap(r) for r in ranks]
        result = np.mean(aps)
    if metric == 'p1':
        hits = [any(r == 1) for r in ranks]
        result = np.mean(hits)
    if metric == 'r1':
        result = recallk(ranks, groundtruth, 1)
    if metric == 'r5':
        result = recallk(ranks, groundtruth, 5)
    return round(result, ndigits=3)


def rank(x):
    temp = x.argsort()
    ranks = np.zeros(len(x))
    ranks[temp] = np.arange(len(x))
    return ranks


def ap(ranks):
    if len(ranks) == 0:
        print 'Query with no relevant candidates associated... nan returned'
        return np.nan
    else:
        recall = np.arange(len(ranks)) + 1
        return np.mean(recall / ranks)


def recallk(ranks, groundtruth, k):
    hits = [sum(r <= k) for r in ranks]
    nrel = [sum(gt) for gt in groundtruth]
    return np.mean(np.array(hits) / np.array(nrel))