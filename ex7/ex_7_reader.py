import struct

import numpy as np
from matplotlib import pyplot as plt, animation, cm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def read_binary_data(filename):
    with open(filename, 'rb') as f:
        niters, a, b, omega, s0_0, s0_1, nbuf = struct.unpack('=iddiiii', f.read(4 + 8 + 8 + 4 + 4 + 4 + 4))

        all_times = np.zeros(niters)
        all_states = np.zeros((niters, 2), dtype=np.int32)

        nread = 0
        for i in range(int(np.ceil(niters / nbuf))):
            nleft = min(nbuf, niters - nread)
            times = np.frombuffer(f.read(nleft * 8))
            states0 = np.frombuffer(f.read(nleft * 4), dtype=np.int32)
            states1 = np.frombuffer(f.read(nleft * 4), dtype=np.int32)

            all_times[nread:nread + nleft] = times
            all_states[nread:nread + nleft, 0] = states0
            all_states[nread:nread + nleft, 1] = states1

            nread += nleft

    params = niters, a, b, omega, s0_0, s0_1, nbuf
    return params, all_times, all_states


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=1.0, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


# every n points are reduced with their means (so we reduce the dimension of the data)
def reduce_with_mean(n, points):
    if len(points.shape) == 1:
        return points.reshape(-1, n).mean(axis=1)
    elif len(points.shape) == 2:
        # np.arange(1, 31).reshape(-1, 2).T.reshape(-1, 2, 5).mean(axis=2).reshape(-1, 3).T
        return points.T.reshape(-1, points.shape[1], n).mean(axis=2).reshape(2, -1).T
    raise ValueError()


def plot_data(params, navg, times, states):
    niters, a, b, omega, s0_0, s0_1, nbuf = params
    x = states[:, 0]
    y = states[:, 1]

    plt.plot(times, x, label="X")
    plt.plot(times, y, label="Y")
    plt.legend()
    plt.title(fr"$\Omega$={omega}, " + fr"$X_0={s0_0}$, " + fr"$Y_0={s0_1}$, "
              + f"n={niters}, " + r"$n_{avg}=$" + f"{navg}")
    plt.xlabel("time")
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    # plt.plot(states[:, 0], states[:, 1], marker=None, linewidth=0.5)
    colorline(x, y, linewidth=1, cmap=plt.get_cmap('gist_gray'))
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    fig.colorbar(cm.ScalarMappable(norm=None, cmap=plt.get_cmap('gist_gray')), ax=plt.gca())
    plt.title(fr"$\Omega$={omega}, " + fr"$X_0={s0_0}$, " + fr"$Y_0={s0_1}$, "
              + f"n={niters}, " + r"$n_{avg}=$" + f"{navg}")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.tight_layout()
    plt.show()


def main():
    fnames = ["omega100_iter200000.bin", "omega1000_iter2000000.bin", "omega10000_iter20000000.bin"]
    # fnames = ["omega100_iter200000.bin"]
    for i, fname in enumerate(fnames):
        filename = r"" + "\\" + fname
        params, times, states = read_binary_data(filename)
        print("File Loaded")
        navg = [200, 500, 1000][i]
        print("Shapes before:", times.shape, states.shape)
        times = reduce_with_mean(navg, times)
        states = reduce_with_mean(navg, states)
        print("Shapes after:", times.shape, states.shape)
        plot_data(params, navg, times, states)


main()
