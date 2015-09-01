__author__ = 'Jan'

import song
import numpy as np


class Collection(object):
    """ Collection of songs """

    def __init__(self, verbose=False):
        self.set = []
        self.verbose = verbose

    def add_song(self, newsong):
        self.set.append(newsong)

    def add_range(self, rng):
        for songid in rng:
            if self.verbose:
                print 'adding song ' + str(songid) + ' of ' + str(len(rng)) + '...'
            self.add_song(song.Song('', '', songid, self.verbose))

    def get_song(self, ind):
        return self.set[ind]

    def get_size(self):
        return len(self.set)

    def get_groundtruth(self):
        return groundtruth(self.set, mode='fast')


def groundtruth(collection1, collection2=None, mode='fast'):
    if collection2 is None:
        collection2 = collection1
    n1 = len(collection1)
    n2 = len(collection2)
    if mode == 'fast':      # vectorized, much faster
        cliques1 = map(song.Song.get_cliquehash, collection1)
        cliques2 = map(song.Song.get_cliquehash, collection2)
        v1 = np.array(cliques1).reshape((n1, 1))
        v2 = np.array(cliques2).reshape((1, n2))
        row = np.ones((1, n2))
        col = np.ones((n1, 1))
        groundtruth = v1.dot(row) - col.dot(v2) == 0
    elif mode == 'slow':
        groundtruth = np.zeros((n1, n2))
        for i, songi in enumerate(collection1):
            for j, songj in enumerate(collection2):
                groundtruth[i, j] = 1 if songi.get_cliquehash() == songj.get_cliquehash() else 0
    return groundtruth