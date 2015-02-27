__author__ = 'Jan'


def plot_timeseries(t, x, start=None, stop=None, savefig=True):
    if x.shape[1] == 1:
        plot_1d(t, x, start, stop, savefig=savefig)
    elif x.shape[1] > 1:
        plot_matrix(x[start, stop].T, savefig=savefig)


def plot_1d(t, x, start=None, stop=None, mode='line', savefig=True):
    import matplotlib.pyplot as plt
    if mode == 'bar':
        plt.bar(t[start:stop], x[start:stop])
    elif mode == 'line':
        plt.plot(t[start:stop], x[start:stop])
    else:
        __unknownmodeerror(['bar', 'line'])
    __release_plot(savefig)


def plot_matrix(mat, start=None, stop=None, savefig=True):
    import matplotlib.pyplot as plt
    mat = mat[start:stop]
    xplt = plt.imshow(mat)
    xplt.set_interpolation('nearest')
    xplt.set_cmap('binary')
    __release_plot(savefig)


def plot_3dmatrix(mat, basemarkersize=1, savefig=True):
    import numpy as np
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    lenx, leny, lenz = mat.shape
    matflat = np.ndarray.flatten(mat)
    matnorm = matflat/max(matflat)
    x = np.repeat(np.repeat(np.arange(lenx), leny), lenz)
    y = np.tile(np.repeat(np.arange(lenx), leny), lenz)
    z = np.tile(np.tile(np.arange(lenx), leny), lenz)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markersize = len(matnorm) * matnorm * basemarkersize
    ax.scatter(x, y, z, c=markersize, s=markersize)
    __release_plot(savefig)


def plot_histogram(data, nbins, mode='bar', zerobin=False, savefig=True, verbose=False):
    import numpy as np
    # import matplotlib.pyplot as plt

    data = data.flatten()
    binedges = np.linspace(min(data), max(data), nbins)
    if zerobin:
        eps = 0.00000001
        binedges = np.hstack((-eps, eps, binedges[1:]))      # quick fix to include a 'zero bin'
    hist, binedges = np.histogram(data, binedges)
    bincenters = binedges[:-1]+0.5*np.diff(binedges)
    if verbose:
        print binedges, bincenters, hist
        print '(number of zeros: ' + str(int(sum(data == 0))) + ' of ' + str(len(data)) + ')'
    plot_1d(bincenters, hist, start=None, stop=None, mode=mode, savefig=savefig)
    return bincenters, hist


def __release_plot(savefig=True):
    import matplotlib.pyplot as plt
    if type(savefig) is str:
        plt.savefig(savefig)
    elif savefig:
        plt.savefig('out.png')
    elif not savefig:
        plt.show()
    plt.close()


def __unknownmodeerror(options):
    print 'Unkown mode. Please try one of the following:'
    print options