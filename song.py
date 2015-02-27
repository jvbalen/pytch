# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:39:39 2014
"""

__author__ = 'Jan'

import numpy as np

# Classes


class Song(object):

    """ Class for items in a dataset of songs, for which summary descriptors can computed and compared.

    Instances of Song contain base features and produce song descriptions based on these base features.

    Currently supported base features are chroma and melody (f0),  read from filedir/filename/chrext and
        filedir/filename/melext, respectively. filename is read from line songid in filelist, other parameters are set
        at construction.

    args
      filedir:  directory of audio files
      filelist: path to file containing filenames list
      songid:   index of song in filenames list
      verbose:  set verbose = True for more output (eg. for debugging)

      mp3ext:   extension of the audio file (default: '.mp3')
      chrext:   extension of chroma data file (default: '_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv')
      melext:   extension of melody data file (default: '_vamp_mtg-melodia_melodia_melody.csv')

    song description methods
      get_pitchhist
      get_pitchbihist
      get_chromacorr
      get_chromabihist
      get_harmonisation
      get_pitchtrihist
      get_chromatrihist

    """
    def __init__(self, filedir, filelist, songid, verbose=False,
                 mp3ext='.mp3',
                 chrext='_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv',
                 melext='_vamp_mtg-melodia_melodia_melody.csv'):

        self.verbose = verbose
        self.songid = songid

        self.mp3ext = mp3ext
        self.chrext = chrext
        self.melext = melext

        self.filedir = filedir
        self.filelist = filelist

        self.mp3file = self.get_filename(ext=self.mp3ext)
        self.chrfile = self.get_filename(ext=self.chrext)
        self.melfile = self.get_filename(ext=self.melext)

        from numpy import nan
        self.precomputed_base_features = False
        self.tchr, self.chroma, self.tmel, self.melody = (nan, nan, nan, nan)

    def get_filename(self, ext=''):
        return read_filelist(self.filedir, self.filelist, self.songid, ext)

    def get_clique(self):
        """ override in a child class
        """
        return 'N/A'

    def get_cliquehash(self):
        return hash(self.get_clique())

    ''' Base feature computation
    '''

    def get_chroma(self):
        t, chroma = read_feature(self.chrfile)
        return t, chroma

    def get_melody(self, unvoiced=True):

        hz2midi = lambda f: 69 + 12 * np.log2(abs(f) / 440)

        t, x = read_feature(self.melfile)
        f0 = x[:, 0]
        if unvoiced:
            f0 = abs(f0)
        else:
            f0[f0 < 0] = 0
        pitched = f0 > 0
        melody = f0
        melody[pitched] = hz2midi(f0[pitched])
        return t, melody

    def get_aligned_features(self):
        if not self.precomputed_base_features:
            self.tchr, self.chroma, self.tmel, self.melody = self.align_base_features()
            self.precomputed_base_features = True
        return self.tchr, self.chroma, self.tmel, self.melody

    def align_base_features(self):
        tchr, chroma = self.get_chroma()
        tmel, melody = self.get_melody()
        # print 'MELODY retrieved from GETMELODY'
        #        print melody
        tchr, chroma, tmel, melody = align_features(tchr, chroma, tmel, melody)
        if self.verbose:
            print 'chroma shape:'
            print chroma.shape
            print 'tmel shape:'
            print tmel.shape
        return tchr, chroma, tmel, melody

    def get_melmat(self, aligned=True):
        from scipy.sparse import csr_matrix

        if aligned:
            tchr, chroma, t, melody = self.get_aligned_features()
        # print 'MELODY retrieved from GETALIGNEDFEATURES'
        #            print melody
        else:
            t, melody = self.get_melody()
        melody = np.round(melody)
        pitched = melody > 0
        pitchclass = np.remainder(melody - 69, 12)
        framerate = 1.0/(t[1]-t[0])

        nmel = len(melody)
        if self.verbose:
            print 'nmel:'
            print nmel
        vals = np.ones(nmel)[pitched]
        vals *= 1.0 / framerate
        rows = np.arange(nmel)[pitched]
        cols = pitchclass[pitched]
        if self.verbose:
            print 'min col in melmat constrution:'
            print min(cols)
            print 'max col in melmat constrution:'
            print max(cols)
        melmat = csr_matrix((vals, (rows, cols)), shape=(nmel, 12))
        if self.verbose:
            print 'melmat shape:'
            print melmat.shape
        return t, melmat.todense()

    def get_melstm(self, win=4.0, aligned=True):
        import scipy.signal as dsp

        t, melmat = self.get_melmat(aligned)
        dt = t[1] - t[0]
        nkern = np.round(win / dt)
        if self.verbose:
            print 'win = ' + str(win)
            print 't0, t1 = ' + str(t[0]) + ' ' + str(t[1])
            print 'dt = ' + str(dt)
            print 'nkern = ' + str(nkern)
        kern1 = np.zeros((nkern + 1, 1))
        kern2 = np.ones((nkern, 1))
        kern = np.vstack((kern1, kern2))
        kern *= 1.0 / nkern
        melstm = dsp.convolve2d(melmat, kern, mode='same')
        return t, melstm, melmat

    def get_melfwd(self, win=4.0, aligned=True):
        import scipy.signal as dsp

        t, melmat = self.get_melmat(aligned)
        dt = t[1] - t[0]
        nkern = np.round(win / dt)
        kern1 = np.ones((nkern, 1))
        kern2 = np.zeros((nkern + 1, 1))
        kern = np.vstack((kern1, kern2))
        kern *= 1.0 / nkern
        melfwd = dsp.convolve2d(melmat, kern, mode='same')
        return t, melfwd, melmat

    def get_chromastm(self, win=4.0, aligned=True):
        import scipy.signal as dsp

        t, chroma = read_feature(self.chrfile)
        dt = t[1] - t[0]
        nkern = np.round(win / dt)
        if self.verbose:
            print 'win = ' + str(win)
            print 't0, t1 = ' + str(t[0]) + ' ' + str(t[1])
            print 'dt = ' + str(dt)
            print 'nkern = ' + str(nkern)
        kern1 = np.zeros((nkern + 1, 1))
        kern2 = np.ones((nkern, 1))
        kern = np.vstack((kern1, kern2))
        kern *= 1.0 / nkern
        chromastm = dsp.convolve2d(chroma, kern, mode='same')
        return t, chromastm, chroma

    # Summary feature computation
    # Bugs / problems:
    #   - ...

    def get_pitchhist(self, minpitch, maxpitch, bpo=12):
        t, melody = self.get_melody()

        step = 12.0 / bpo
        halfstep = step / 2.0
        nbins = (maxpitch - minpitch) / step + 1
        binedges = np.linspace(minpitch - halfstep, maxpitch + halfstep, nbins + 1)

        pitchhist = np.histogram(melody, binedges)
        bincenters = (binedges[0:-1] + binedges[1:]) / 2
        return pitchhist[0], bincenters

    def get_pitchbihist(self, win=0.5, aligned=True, diagfactor=0, sqrt=True):
        t, melstm, melmat = self.get_melstm(win=win, aligned=aligned)
        pitchbihist = co_occurrence([melstm, melmat], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            pitchbihist = scale_diag(pitchbihist, diagfactor)
        if sqrt:
            pitchbihist = np.sqrt(pitchbihist)
        return pitchbihist

    def get_chromacorr(self, diagfactor=0.5, mode='corr'):
        tchr, chroma, tmel, melody = self.get_aligned_features()
        chromacorr = co_occurrence([chroma], mode=mode, verbose=self.verbose)
        if diagfactor < 1:
            chromacorr = scale_diag(chromacorr, diagfactor)
        return chromacorr

    def get_chromabihist(self, win=0.5, diagfactor=0.5):
        t, chromastm, chroma = self.get_chromastm(win=win)
        chromabihist = co_occurrence([chromastm, chroma], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            chromabihist = scale_diag(chromabihist, diagfactor)
        return chromabihist

    def get_harmonisation(self, diagfactor=0):
        tchr, chroma, tmel, melody = self.get_aligned_features()
        dt = tchr[1] - tchr[0]
        chroma = chroma[2:, :] * dt     # cropping is for exact matlab correspondence
        t, melmat = self.get_melmat(aligned=True)
        melmat = np.array(melmat)
        melmat = melmat[2:, :]          # cropping is for exact matlab correspondence
        if self.verbose:
            print 'melmat and chroma types and shapes:'
            print type(melmat)
            print type(chroma)
            print melmat.shape
            print chroma.shape
        harmonisation = co_occurrence([melmat, chroma], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            harmonisation = scale_diag(harmonisation, diagfactor)
        return harmonisation

    def get_pitchtrihist(self, win=0.5, diagfactor=0, norm=True):
        t, melstm, melmat = self.get_melstm(win, aligned=True)
        t, melfwd, melmat = self.get_melfwd(win, aligned=True)
        pitchtrihist = co_occurrence([melstm, melmat, melfwd], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            pitchtrihist = scale_diag(pitchtrihist, diagfactor)
        if norm:
            pitchtrihist = pitchtrihist * 1.0 / np.sum(pitchtrihist)
        return pitchtrihist


    def get_chromatrihist(self, win=0.5, diagfactor=0, norm=True):
        t, melstm, melmat = self.get_melstm(win, aligned=True)
        tchr, chroma, tmel, melody = self.get_aligned_features()
        chromatrihist = co_occurrence([melstm, melmat, chroma], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            chromatrihist = scale_diag(chromatrihist, diagfactor)
        if norm:
            chromatrihist = chromatrihist * 1.0 / np.sum(chromatrihist)
        return chromatrihist

    def get_xgram(self, win=0.5, diagfactor=0, mode='conf'):
        t, melstm, melmat = self.get_melstm(win, aligned=True)
        t, melfwd, melmat = self.get_melfwd(win, aligned=True)
        tsplit = np.round(len(t)/2.0)
        t1, melstm1, melmat1, melfwd1 = t[:tsplit], melstm[:tsplit], melmat[:tsplit], melfwd[:tsplit]
        t2, melstm2, melmat2, melfwd2 = t[tsplit:], melstm[tsplit:], melmat[tsplit:], melfwd[tsplit:]
        pitchtrihist1 = co_occurrence([melstm1, melmat1, melfwd1], mode='dot', verbose=self.verbose)
        pitchtrihist2 = co_occurrence([melstm2, melmat2, melfwd2], mode='dot', verbose=self.verbose)
        if diagfactor < 1:
            pitchtrihist1 = scale_diag(pitchtrihist1, diagfactor)
            pitchtrihist2 = scale_diag(pitchtrihist2, diagfactor)
        xgram = combine_hist(pitchtrihist1, pitchtrihist2, mode=mode)
        return xgram


# Support functions


def read_filelist(filedir, filelist, listindex, ext):
    import csv

    filelistfile = open(filelist)
    listreader = csv.reader(filelistfile)
    listentries = list(listreader)[listindex]
    return filedir + listentries[0] + ext


def read_feature(filename, mode='pandas'):
    import numpy as np
    import pandas as pd

    if mode == 'numpy':
        data = np.genfromtxt(filename, delimiter=',')
    elif mode == 'pandas':
        data = pd.read_csv(filename, delimiter=',').values
    t = data[:, 0]
    x = data[:, 1:]
    return t, x


def align_features_old(t1, x1, t2, x2, verbose=False):
    # Obsolete: digitize took up most of the experiment runtime under this implementation.
    #   from cProfile, pitchfps experiment with n = 5:
    #     176551 function calls (172944 primitive calls) in 7.906 seconds
    #     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #        ...
    #         20    0.002    0.000    1.213    0.061 song.py:281(read_feature)
    #         10    0.006    0.001    6.495    0.649 song.py:294(align_features)
    #        ...
    # New version above exploits constant framerate.
    import numpy as np
    dt1 = t1[1] - t1[0]
    dt2 = t2[1] - t2[0]
    if dt1 > dt2:
        edges = t2[:-1] + 0.5 * np.diff(t2)
        ind = np.digitize(t1, edges)
        if verbose: print 'Warning: t2 discarded during alignment (case dt1 > dt2).'
        t2 = t1
        x2 = x2[ind]
    else:
        edges = t1[:-1] + 0.5 * np.diff(t1)
        ind = np.digitize(t2, edges)
        if verbose: print 'Warning: t1 discarded during alignment (case dt2 > dt1).'
        t1 = t2
        x1 = x1[ind]
    return t1, x1, t2, x2


def align_features(tx, x, ty, y):
    """ 'Conservative' alignment of time series tx, x and tx, y:
            First, both time series are cropped to the range where they are both defined.
            The time series with highest resolution is then downsampled to obtain the same number of elements for both.
            This is done by retaining the samples closest to those of the other time series.
            For time values of the downsampled time series, the nearest time values of the reference series are used.
    """
    import numpy as np
    dtx = tx[1]-tx[0]
    dty = ty[1]-ty[0]
    xbegins = x[0] <= y[0]
    xends = x[-1] >= y[-1]
    if dtx >= dty:
        # find subset of indices of y closest to x
        iresample = np.round((tx-ty[0])/dty).astype(int)
        icrop = (0 <= iresample) * (iresample < y.size)
        y = y[iresample[icrop]]
        x = x[icrop]
        tx = tx[icrop]
        ty = tx
    elif dty > dtx:
        # find subset of indices of x closest to y
        iresample = np.round((ty-tx[0])/dtx).astype(int)
        icrop = (0 <= iresample) * (iresample < x.size)
        x = x[iresample[icrop]]
        y = y[icrop]
        ty = ty[icrop]
        tx = ty
    return tx, x, ty, y

    # start = max(min(x[0],y[0]))
    # end = min(max(x[-1],y[-1]))


def co_occurrence(mats, mode='dot', verbose=False):
    import numpy as np
    from scipy import einsum
    if len(mats) == 1:
        x = mats[0]
        if verbose:
            print 'type of x in co-occurrence computation: ' + str(type(x))
        if mode == 'dot':
            cont = x.T.dot(x)  # equivalent of x' * y
        elif mode == 'normdot':
            x /= np.linalg.norm(x, axis=0)
            cont = x.T.dot(x)
        elif mode == 'posdot':
            x -= np.mean(x, axis=0)
            x /= np.std(x, axis=0)
            x[x < 0] = 0        # without this line, equivalent to 'corr'
            cont = x.T.dot(x)
        elif mode == 'corr':
            cont = np.corrcoef(x, rowvar=0)
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            cont = x.T.dot(x)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')
    if len(mats) == 2:
        x, y = mats[:]
        if mode == 'dot':
            cont = x.T.dot(y)  # equivalent of x' * y
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            y = np.sparse.csr_matrix(y)
            cont = x.T.dot(y)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')
    if len(mats) == 3:
        x, y, z = mats[:]
        if mode == 'dot':
            cont = einsum('ni,nj,nk', x, y, z)
        elif mode == 'corr':
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
            z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
            cont = einsum('ni,nj,nk', x, y, z)
        else:
            raise NameError('mode does not exist')
    return cont


def scale_diag(x, diag_factor):
    # TODO: scale not only z[i,i,i] but all of x[i,i,:] and x[:,i,i]
    # More elegant seeing how the trigrams are computed (from the middle pitch out).
    import numpy as np
    diag = get_diag(x)
    shp = x.shape
    x = np.array(x).flatten()
    x[diag] = diag_factor * x[diag]
    return x.reshape(shp)


def get_diag(x):
    import numpy as np

    shp = x.shape
    ndim = len(shp)
    diag = np.zeros(shp)
    if ndim == 1:
        diag = 0
    elif ndim == 2:
        i = np.arange(min(shp))
        ii = (i, i)
        diag = np.ravel_multi_index(ii, shp)
    elif ndim == 3:
        i = np.arange(min(shp))
        iii = (i, i, i)
        diag = np.ravel_multi_index(iii, shp)
    return diag


def combine_hist(h1, h2, mode='min', alpha=1, verbose=False):
    """ Combine information from two histograms h1 and h2 into a single histogram h, according to parameter mode
    mode == 'min' uses the minimum of h1 and h2
    mode == 'mean' uses the mean of h1 and h2
    mode == 'prod' uses the product of h1 and h2
    mode == 'geom' uses the geometric mean of h1 and h2 (sqrt of product)
    mode == 'conf' uses the lower bound of a prediction interval given 2 observations h1 and h2, and an alpha (in %)
        formula assumes a lognormal distribution which makes:
        prediction interval = A (h1 + h2) +/- B (h1 - h2)
        where   A = 0.5
                B = z_alpha / sqrt(2)
        alpha defaults to 1%
    mode == 'log' is like 'conf' but assumes a lognormal distribution which makes:
        prediction interval = EXP[ A log(h1 * h2) +/- B log(h1 / h2) ]
    NOTE: since h1, h2, h are histograms, negative values for h will be adjusted to zero
    """
    import numpy as np
    z = {0.5: 2.58, 1: 2.33, 2.5: 1.96, 5: 1.65, 10: 1.28, 25: 0.68, 50: 0,
         75: -0.68, 90: -1.28, 95: -1.65, 97.5: -1.96, 99: -2.33, 99.5: -2.58}
    A, B = 0.50, 0.71 * z[alpha]
    if mode == 'min':
        h = np.minimum(h1, h2)
    elif mode == 'mean':
        h = (h1 + h2) * 0.5
    elif mode[:4] == 'prod':
        h = h1 * h2
    elif mode[:4] == 'geom':
        h = np.sqrt(h1 * h2)
    elif mode[:4] == 'conf':
        # assuming normal distribution
        h = A * (h1 + h2) - B * np.abs(h1 - h2)
    elif mode == 'log':
        # assuming log-normal distribution
        lim = A * np.log(h1 * h2) - B * np.abs(np.log(h1) - np.log(h2))
        h = np.exp(lim)
    else:
        h = h1
    if verbose:
        print 'mode: ' + mode
        print 'n_nan = {}'.format(np.sum(np.isnan(h)))
        print 'n_negative = {}'.format(np.sum(h < 0))
        print 'n_zero = {}'.format(np.sum(h == 0))
    h[np.isnan(h)] = 0
    h[h < 0] = 0
    return h