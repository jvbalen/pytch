__author__ = 'Jan'

import song
import collection
import experiment
import fingerprint as fp
import numpy as np


class C80Song(song.Song):
    """ Covers80 Song Class

    Class for items in the Covers80 dataset of cover songs.
    This dataset is available at http://labrosa.ee.columbia.edu/ projects/coversongs/covers80/.
    Set filedir and filelist to your local path to this dataset to run experiments.

    TODO: make local links relative or generic.
    """

    def __init__(self, songid, verbose=False):
        filedir = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'
        filelist = '/Users/Jan/Documents/Work/Cogitch/Audio/covers80/'

        mp3ext = '.mp3'
        chrext = '_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv'
        melext = '_vamp_mtg-melodia_melodia_melody.csv'
        song.Song.__init__(self, filedir, filelist, songid, verbose=verbose,
                      mp3ext=mp3ext, chrext=chrext, melext=melext)

    def get_filename(self, ext=''):

        if 0 <= self.songid < 80:
            newfilelist = self.filelist + 'list1.list'
            listindex = self.songid
        elif 80 <= self.songid < 160:
            newfilelist = self.filelist + 'list2.list'
            listindex = self.songid - 80
        else:
            print 'Song failed to load: songid must be in range 0..159.'

        return song.read_filelist(self.filedir, newfilelist, listindex, ext)

    def get_clique(self):
        return np.remainder(self.songid, 80)


class C80Collection(collection.Collection):
    """ Collection representation of the Covers80 dataset"""

    def __init__(self, rng=range(160), verbose=False):
        collection.Collection.__init__(self,verbose)
        self.add_range(rng)

    def add_range(self, rng=range(160)):
        for songid in rng:
            if self.verbose:
                print 'adding song ' + str(songid) + ' of ' + str(len(rng)) + '...'
            self.add_song(C80Song(songid, self.verbose))


class C80Experiment(experiment.Experiment):
    def __init__(self, rng=range(80), fptype=fp.mixedfp, params=None):
        if not params: params = {}

        queries = C80Collection(rng)
        candidates = C80Collection(list(np.array(rng) + 80))
        groundtruth = collection.collection_groundtruth(queries.collection, candidates.collection)
        experiment.Experiment.__init__(self, queries, candidates, groundtruth, fptype, params)