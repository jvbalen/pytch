__author__ = 'Jan'

import song
import collection
import experiment
import plotting
import fingerprint as fp
import numpy as np

class T2KSong(song.Song):
    """ T2K Cover Song Class

    Class for items in the (unreleased) T2K dataset of cover songs.
    """

    def __init__(self, songid, verbose=False):
        filedir = '/Users/Jan/Documents/Work/Cogitch/Data/T2K/'
        filelist = '/Users/Jan/Documents/Work/Cogitch/Data/T2K/chroma.txt'

        mp3ext = ''
        chrext = ''
        melext = ''
        song.Song.__init__(self, filedir, filelist, songid, verbose=verbose,
                           mp3ext=mp3ext, chrext=chrext, melext=melext)
        self.cliquelist = '/Users/Jan/Documents/Work/Cogitch/Data/T2K/chroma.txt'

    def get_filename(self, ext=''):
        return song.read_filelist(self.filedir, self.filelist, self.songid, ext)

    def get_spotifyid(self):
        filename = self.get_filename()
        filename = filename.split('/')[-1]
        return filename.split('.')[0]

    def get_clique(self):
        filename = self.get_filename()
        return filename.split('/')[-3]

    def get_cliquehash(self):
        return hash(self.get_clique())

    def get_chroma(self):
        if not self.precomputed_base_features:
            # print 'reading chroma for song ' + self.get_filename()
            import pandas as pd

            filename = self.get_filename(ext=self.chrext)
            data = pd.read_csv(filename, delimiter=',').values
            tchr = 0.1 * np.arange(data.shape[0])   # PLACE-HOLDER
            chr = data[:, :12]
        else:
            tchr, chr, tmel, mel = self.get_aligned_features()
        return tchr, chr

    def get_melody(self):
        tchr, chr = self.get_chroma()
        return tchr, np.argmax(chr, axis=1).reshape(chr.shape[0], 1)

    def get_melmat(self, aligned=True):
        from scipy.sparse import csr_matrix
        tchr, chr, tmel, mel = self.get_aligned_features()
        mx = np.amax(chr, axis=1).reshape(chr.shape[0], 1)
        thr = mx.dot(np.ones((1, chr.shape[1])))
        melmat = chr * (chr >= thr)
        melmat = csr_matrix(melmat)
        return tchr, melmat.todense()

    def align_base_features(self):
        tchr, chroma = self.get_chroma()
        tmel, melody = self.get_melody()
        return tchr, chroma, tmel, melody

    def get_aligned_features(self):
        if not self.precomputed_base_features:
            self.tchr, self.chroma, self.tmel, self.melody = self.align_base_features()
            self.precomputed_base_features = True
        return self.tchr, self.chroma, self.tmel, self.melody


class T2KCorpus(collection.Collection):
    """ Collection representation of the Covers80 dataset"""

    def __init__(self, rng=range(1536), verbose=False):
        collection.Collection.__init__(self,verbose)
        self.add_range(rng)

    def add_range(self, rng=range(1536)):
        for songid in rng:
            if self.verbose:
                print 'adding song ' + str(songid) + ' of ' + str(len(rng)) + '...'
            self.add_song(T2KSong(songid, self.verbose))

    def get_cliquememberids(self):
        cliqueids = np.array(map(T2KSong.get_cliquehash, self.collection))
        spotifyids = np.array(map(T2KSong.get_spotifyid, self.collection))
        cliquememberids = np.zeros(self.get_size())
        cliquesize = np.zeros(self.get_size())
        for i in range(self.get_size()):
            member = cliqueids == cliqueids[i]
            smaller = spotifyids < spotifyids[i]
            memberandsmaller = np.all(np.vstack((member, smaller)), axis=0)
            cliquememberids[i] = np.sum(memberandsmaller)
            cliquesize[i] = np.sum(member)
        return cliquememberids, cliquesize


class T2KExperiment(experiment.Experiment):

    def __init__(self, rng=range(1535), fptype=fp.mixedfp, params={'weights': (2, 1, 0)}):
        fullcorp = T2KCorpus(rng)
        memberid, cliquesize = fullcorp.get_cliquememberids()
        qrange = np.array(rng)[(cliquesize > 1) * (memberid == 0)]
        crange = np.array(rng)[(cliquesize > 1) * (memberid != 0)]

        queries = T2KCorpus(qrange)
        candidates = T2KCorpus(crange)
        groundtruth = collection.collection_groundtruth(queries.collection, candidates.collection)
        experiment.Experiment.__init__(self, queries, candidates, groundtruth, fptype, params)


def optimize_T2Kfingerprinting(n=5):
    exp = T2KExperiment()
    print 'Initiation with default weights...'
    exp.run(distmetric='cosine', nmatches=1, evalmetrics=('map', 'p1', 'r5'), verbose=False)
    print 'Results: ' + str(exp.results)
    for w0 in range(n):
        for w1 in range(n):
            for w2 in range(n):
                w = (w0, w1, w2)
                exp.recompute_fps(weights=w)
                exp.run(verbose=False)
                print 'w = ' + str(w) + '             ' + str(exp.results)


def test_T2KSong():

    # id = 208
    # s = T2KSong(id)

    # print s.get_filename()
    # print s.get_spotifyid()
    # print s.get_clique()

    # tchr, chr = s.get_chroma()
    # print tchr.shape
    # print tchr[:20]
    # print chr.shape
    # print chr[:20, :]

    # tmel, mel = s.get_melody()
    # print tmel.shape
    # print tmel[:20]
    # print mel.shape
    # print mel
    # print mel[:20, :]

    # pbh = s.get_pitchbihist(win=1)
    # print pbh
    # plotting.plot_matrix(pbh)

    # ccc = s.get_chromacorr()
    # print ccc
    # plotting.plot_matrix(ccc)

    # h = s.get_harmonisation()
    # print h
    # plotting.plot_matrix(h)

    # c = T2KCorpus(rng=range(1537))      # nmax = 1535
    # print c.get_size()
    # plotting.plot_matrix(c.get_groundtruth())

    # c = T2KCorpus(rng=range(100))      # nmax = 1535
    # print c.get_size()
    # print np.max(np.abs(c.get_groundtruth(mode='fast') - c.get_groundtruth(mode='slow')))

    # c = T2KCorpus(rng=range(100))
    # print c.get_cliquememberids()

    e = T2KExperiment(rng=range(1535))
    print 'e running..'
    e.run(verbose=True)

    # optimize_T2Kfingerprinting(n=5)

if __name__ == '__main__':
    import timeit
    t = timeit.Timer('test_T2KSong()', 'from __main__ import test_T2KSong')
    print t.timeit(number=1)