__author__ = 'Jan'

import song
import collection
import fingerprint
import experiment
import plotting
import numpy as np
import numpy.random as rnd


def test_alignment(verbose=True):
    t1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    t2 = np.array([0, 5, 10, 15, 20])
    x1 = rnd.rand(len(t1))
    x2 = rnd.rand(len(t2))
    ta1, xa1, ta2, xa2 = song.align_features(t1, x1, t2, x2)
    if verbose:
        print t1
        print t2
        print ta1
        print ta2


def test_chromacorr(mode='corr'):
    s = song.C80Song(0)
    ccc = s.get_chromacorr(mode=mode)
    print ccc


def test_pitchtrihist(verbose=False):
    s = song.C80Song(0)
    pth = s.get_pitchtrihist()
    h, edges = np.histogram(pth, bins=30)
    if verbose:
        print 'edges: ' + str(edges)
        print 'hist: ' + str(h)
        print 'PITCHTRIHIST:'
        print pth
    plotting.plot_3dmatrix(pth)


def test_scalediag(n=4, verbose=True):

    a = 10 * np.ones((n, n, n))
    # a = 10 * np.ones((n, n))
    # a = 10 * np.ones(n)
    scaled = song.scale_diag(a, 0.5)
    if verbose:
        print 'a = '
        print a
        print 'scaled a = '
        print scaled


def test_songclass(songid, verbose=False):
    s = song.C80Song(songid)
    s.verbose = verbose

    # hist, bins = s.get_pitchhist()
    # plotting.plot_1d(bins, hist)

    # t, x = s.get_melody(-1100,-1000,True)
    # plotting.plot_timeseries(t, x)

    # m = s.get_chroma(-1100,-1000,True)
    # m = s.get_melmat(-1100,-1000,True)
    # m = s.get_melstm(.5,-1100,-1000,True)
    # m = s.get_pitchbihist(diagfactor=0)
    # m = s.get_chromacorr()
    # m = s.get_harmonisation()
    # plotting.plot_matrix(m, savefig=True)

    # m = s.get_pitchtrihist(diagfactor=0)
    m = s.get_chromatrihist()
    plotting.plot_3dmatrix(m, savefig=True)


def test_corpusclass(verbose=False):
    c = collection.C80Corpus(range(10), verbose)
    f1 = fingerprint.Table(c, fptype=fingerprint.pbhfp, params={})
    f2 = [fingerprint.pbhfp(s, win=0.5) for s in c.collection]
    print f1
    print f2


def test_fingerprints(verbose=False, makehist=True):
    s = song.C80Song(10)
    fp = fingerprint.pitchfp(s)
    # fp = fingerprint.threshold_fp(fp, 0.5, mode='tops')
    if makehist:
        nbins = 30
        plotting.plot_histogram(fp, nbins, mode='line', zerobin=True, savefig=True, verbose=verbose)
    else:
        plotting.plot_3dmatrix(1.0 / 43 * fp.reshape((12, 12, 12)))


def test_evaluate():
    # d = np.array([[0,0],[0,0]])
    # g = np.array([[1,0],[0,1]])

    # d = np.zeros((10,10))
    # d = np.ones((10,10))
    # d = rnd.rand(10,10)
    # g = np.eye(10)
    # g = np.zeros((10,10))
    # g = np.ones((10,10))

    d = np.ones(10) - np.eye(10)
    d[0, 0] = 0.5
    d[0, 1] = 0
    print(d)
    g = np.eye(10)
    import experiment

    print experiment.evaluate(d, g)


def test_keyinvariant():
    n = 3
    # a = rnd.rand(n, n, n)
    a = rnd.rand(n, n)
    print 'a: '
    print a

    nshift = 1
    b = a.copy()
    b = np.roll(b, nshift, axis=0)
    b = np.roll(b, nshift, axis=1)
    # b = np.roll(b, nshift, axis=2)
    print 'b: '
    print b

    ainv = fingerprint.all_keys(a)
    print 'ainv:'
    print ainv

    binv = fingerprint.all_keys(b)
    print 'binv: '
    print binv

    print 'difference: '
    print abs(ainv - binv)


def test_distances():
    n = 10
    qdb = fingerprint.Table(collection.Collection(), fingerprint.mixedfp, {})
    qdb.fps = rnd.rand(n, 3)
    qdb.ids = list(np.arange(n) / 2.0)
    cdb = fingerprint.Table(collection.Collection(), fingerprint.mixedfp, {})
    cdb.fps = rnd.rand(n, 3)
    cdb.ids = list(np.arange(n) / 2.0)
    exp = experiment.TestExperiment(qdb, cdb)

    fpdm = exp.fp_distances()
    songdm, qids, cids = exp.song_distances(nmatches=1)

    print 'fp DM'
    print fpdm
    print 'song DM'
    print songdm


def test_keyinvar(n=80, fptype=fingerprint.pitchfp, params=None):
    if not params:
        params = {'keyhandling': 'keyinvar'}
    exp = experiment.C80Experiment(range(n), fptype, params)

    exp.run(distmetric='cosine', nmatches=1, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 1: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=2, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 2: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=3, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 3: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=6, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 6: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=12, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 12: \t' + str(exp.results)

    exp.run(distmetric='cosine', nmatches=13, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 13: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=15, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 15: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=30, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 30: \t' + str(exp.results)
    exp.run(distmetric='cosine', nmatches=60, evalmetrics=('map', 'p1', 'r5'))
    print 'nmatches = 60: \t' + str(exp.results)


def test_matlab01():
    songid = 0
    s = song.C80Song(songid)
    tchr, chroma, tmel, melody = s.get_aligned_features()
    ph = np.sum(chroma, 0)
    ph = ph / np.sum(sum(ph))
    print ph

    c1 = s.get_chromacorr(diagfactor=0)
    c1 = fingerprint.flatten_feature(c1, keyhandling='none')
    c1 = fingerprint.concat_features([c1], normfunction='whiten')
    print c1.reshape(12, 12)[:5, :5]

    params = {'normfunction': 'whiten'}
    exp = experiment.C80Experiment(range(5), fptype=fingerprint.mixedfp, params=params)
    exp.run(verbose=False)
    c2 = exp.qdb.fps[songid]
    c2 = c2[:144]
    print c2.reshape(12, 12)[:5, :5]


def test_matlab03(n=5):
    # params = {'normfunction': 'whiten', 'keyhandling': 'trans'}
    exp = experiment.C80Experiment(range(n), fptype=fingerprint.cccfp)
    exp.run(verbose=False)
    d = exp.fp_distances()
    print d


def test_matlab04(n=80):
    params = {'diagfactor': 0.5, 'normfunction': 'whiten', 'keyhandling': 'trans'}
    exp = experiment.C80Experiment(range(n), fptype=fingerprint.cccfp, params=params)
    exp.run(verbose=False)
    print exp.results
    # matches results from matlab's test_python04 AND the paper, with diagfactor set correctly (0.5)


def test_matlab05():
    songid = 0
    s = song.C80Song(songid)
    # t, melmat = s.get_melmat(aligned=False)
    # print melmat[-10:, :].todense()
    t, melstm, melmat = s.get_melstm(win=0.5, aligned=False)
    print melstm[-10:, :]
    pass


def test_matlab06():
    songid = 0
    s = song.C80Song(songid)
    pbh = s.get_pitchbihist(win=0.5, aligned=False, diagfactor=0, sqrt=True)
    print pbh


def test_matlab07(n=80):
    params = {'win': 0.5, 'diagfactor': 0, 'normfunction': 'whiten', 'keyhandling': 'trans'}
    exp = experiment.C80Experiment(range(n), fptype=fingerprint.pbhfp, params=params)
    exp.run(verbose=False)
    print exp.results


def test_matlab08():
    songid = 0
    s = song.C80Song(songid)
    pbh = s.get_harmonisation(diagfactor=1)
    print pbh


def test_matlab09(n=80):
    params = {'diagfactor': 1, 'normfunction': 'whiten', 'keyhandling': 'trans'}
    exp = experiment.C80Experiment(range(n), fptype=fingerprint.harmfp, params=params)
    exp.run(verbose=False)
    print exp.results


def test_matlab10(n=80):
    params = {'win': 0.5, 'diagfactor': 0.5, 'normfunction': 'whiten', 'keyhandling': 'trans'}
    exp = experiment.C80Experiment(range(n), fptype=fingerprint.cbhfp, params=params)
    exp.run(verbose=False)
    print exp.results


def test_xgram():
    songid = 0
    s = song.C80Song(songid)
    x = s.get_xgram(win=0.5, diagfactor=1, mode='prod')
    print 'sparsity =  {}'.format(np.mean(x == 0))


def test_experiment(n=80, fptype=fingerprint.mixedfp, params=None):
    if params is None:
        params = {'weights': (2, 3, 1)}
    exp = experiment.C80Experiment(range(n), fptype=fptype, params=params)
    exp.run(distmetric='cosine', nmatches=1, evalmetrics=('map', 'p1', 'r5'), verbose=True)
    pass # for debuggability


def runtests():
    # test_alignment()
    # test_pitchtrihist(verbose=True)
    # test_scalediag()
    # test_songclass(0, verbose=True)
    # test_corpusclass(verbose=False)
    # test_fingerprints()
    # test_evaluate()
    # test_distances()

    # test_matlab01()
    # test_matlab03()
    # test_matlab04(n=80)
    # test_matlab05()
    # test_matlab06()
    # test_matlab07(n=80)
    # test_matlab08()
    # test_matlab09(n=80)
    # test_matlab10(n=80)

    # test_experiment(n=5)
    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (1, 0, 0)})
    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (0, 1, 0)})
    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (0, 0, 1)})
    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (2, 3, 1)})
    test_experiment(n=80, fptype=fingerprint.pitchfp, params={})
    # test_experiment(n=80, fptype=fingerprint.chromafp, params={})

    # test_keyinvar(5)
    # test_keyinvar(n=80, fptype=fingerprint.pitchfp)
    # test_keyinvar(n=80, fptype=fingerprint.chromafp)

    # test_xgram()
    # test_experiment(n=80, fptype=fingerprint.pitchfp, params={})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'min'})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'mean'})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'prod'})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'geom'})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'conf'})
    # test_experiment(n=80, fptype=fingerprint.xgramfp, params={'mode': 'log'})

    # test_chromacorr(mode='dot')
    # test_chromacorr(mode='corr')
    # test_chromacorr(mode='normdot')

    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (1, 0, 0)})
    # test_experiment(n=80, fptype=fingerprint.cccfp, params={'mode': 'dot'})
    # test_experiment(n=80, fptype=fingerprint.cccfp, params={'mode': 'corr'})
    # test_experiment(n=80, fptype=fingerprint.cccfp, params={'mode': 'normdot'})
    # test_experiment(n=80, fptype=fingerprint.cccfp, params={'mode': 'posdot'})

    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (2, 3, 1)})
    # test_experiment(n=80, fptype=fingerprint.mixedfp, params={'weights': (0, 1, 0)})

    print 'Done testing.'


if __name__ == '__main__':
    # import cProfile, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()

    runtests()

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()