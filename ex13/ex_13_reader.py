import struct

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def read_binary_data(fname_stats):
    datas = []
    with (open(fname_stats, 'rb') as f):
        nsim = np.frombuffer(f.read(4), dtype=np.int32)[0]

        for k in range(nsim):
            N, niters, dt, cellside, ncellside, side, temp, eps, ncell = struct.unpack("=iiddidddi",
                                                                                       f.read(4 * 4 + 8 * 5))

            r1, r2, D1, D2, mu1, mu2, density = struct.unpack("=ddddddd", f.read(8 * 7))

            params = N, niters, dt, cellside, ncellside, side, temp, eps, ncell, r1, r2, D1, D2, mu1, mu2, density
            types = np.frombuffer(f.read(N * 4), dtype=np.int32)

            out = np.frombuffer(f.read(2 * N * niters * 8)).reshape(niters, N, 2)

            # data = np.empty((niters, N, 2))
            # data[:, :, 0] = out[:, :, ::2]
            # data[:, :, 1] = out[:, :, 1::2]
            datas.append((params, types, out))

    return datas


def animate_particles(data, types, side, skip=1):
    num_frames = data.shape[0] // skip
    num_particles = data.shape[1]

    fig, ax = plt.subplots()
    # max_val = np.max(np.abs(data[:, :, 1:]))
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect('equal')

    colors = ["C0" if t == 1 else "C1" for t in types]
    scatter = ax.scatter(data[0, :, 0], data[0, :, 1], c=colors, s=70)
    # tails = [ax.plot([], [], color=colors[i], lw=2, zorder=-1)[0] for i in range(num_particles)]

    def update(frame):
        scatter.set_offsets(data[frame * skip, :, :])
        # start = max(0, frame - 20)
        # for i, tail in enumerate(tails):
        #     tail.set_xdata(data[start:frame, i, 0])
        #     tail.set_ydata(data[start:frame, i, 1])
        # return scatter, *tails
        return scatter,

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
    plt.show()

def plot_msd_over_time(params, types, data):
    dt = params[2]
    niters = params[1]

    plt.figure()

    only1_flt = types == 1
    only2_flt = types == 2

    y1 = (data[:, only1_flt, 0] - data[0, only1_flt, 0]) ** 2
    y2 = (data[:, only2_flt, 0] - data[0, only2_flt, 0]) ** 2
    time = np.arange(niters) * dt
    plt.plot(time, y1.mean(axis=1))
    plt.plot(time, y2.mean(axis=1))
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def main():
    bname = r"\\wsl.localhost\Ubuntu\home\emanuele\nmsm\ex13"
    fname = bname + "\\" + "stats_ex11_N200_M200000.bin"
    datas = read_binary_data(fname)
    # animate_particles(datas[0][2], datas[0][1], 100, skip=50)
    plot_msd_over_time(datas[0][0], datas[0][1], datas[0][2])
main()
