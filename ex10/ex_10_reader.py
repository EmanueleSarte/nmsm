import struct

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import maxwell


def read_binary_data(fname_stats, fname_particles=None):
    datas = []
    with open(fname_stats, 'rb') as f:
        nsim = np.frombuffer(f.read(4), dtype=np.int32)[0]

        for k in range(nsim):
            N, niters, nbuf, save_par_every, dt, eps, sigma, rho, sizex, sizey, sizez, sigma_cut, omega, thermostat = struct.unpack(
                "=iiiidddddddddi", f.read(4 * 5 + 8 * 9))

            stats = np.zeros((niters, 3))
            nread = 0
            for i in range(int(np.ceil(niters / nbuf))):
                nleft = min(nbuf, niters - nread)
                stats[nread:nread + nleft, 0] = np.frombuffer(f.read(nleft * 8))
                stats[nread:nread + nleft, 1] = np.frombuffer(f.read(nleft * 8))
                stats[nread:nread + nleft, 2] = np.frombuffer(f.read(nleft * 8))

                nread += nleft

            gnum, glen, dxg = struct.unpack("=iid", f.read(4 + 4 + 8))
            gpair_val = np.frombuffer(f.read(gnum * glen * 8)).reshape(gnum, glen)

            params = N, niters, nbuf, save_par_every, eps, sigma, rho, sizex, sizey, sizez, sigma_cut
            datas.append([params, stats, (dxg, gpair_val)])

    if fname_particles:
        try:
            with open(fname_particles, "rb") as f:
                for k in range(nsim):
                    N = datas[k][0][0]
                    M = datas[k][0][1] // datas[k][0][3]  # niters / save_par_particle

                    out = np.frombuffer(f.read(6 * N * M * 8)).reshape(M, 2, N * 3)

                    data = np.empty((M, N, 6))
                    data[:, :, 0] = out[:, 0, ::3]
                    data[:, :, 1] = out[:, 0, 1::3]
                    data[:, :, 2] = out[:, 0, 2::3]
                    data[:, :, 3] = out[:, 1, ::3]
                    data[:, :, 4] = out[:, 1, 1::3]
                    data[:, :, 5] = out[:, 1, 2::3]
                    datas[k].append(data)

        except Exception as e:
            for k in range(nsim):
                datas[k].append(None)
    else:
        for k in range(nsim):
            datas[k].append(None)

    return datas


def animate_particles_3d(params, data):
    num_frames, num_particles, _ = data.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot(data[0, :, 0], data[0, :, 1], data[0, :, 2], marker='o', linewidth=0)

    def update(frame):
        # scatter._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        # scatter.set_data_3d(data[frame, :, :])
        line.set_data(data[frame, :, 0:2].swapaxes(0, 1))
        line.set_3d_properties(data[frame, :, 2])
        ax.set_title(f"Time Step: {frame}")
        return line,  # Important: return the artist for blitting

    # Set axis limits (you might need to adjust these based on your data)
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    z_min, z_max = 0, 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True)  # blit=True for faster animation

    plt.show()


def plot_pair_corr(datas):
    npdata = None
    for i, (params, _, (dxg, gcorr), _) in enumerate(datas):
        if npdata is None:
            npdata = np.zeros((len(datas), gcorr.shape[-1]))
        npdata[i] = np.mean(gcorr, axis=0)

    rows = 2
    cols = 8
    # xmin, xmax = 0.75, 1.75
    xmin, xmax = 1.5, 4
    ymin, ymax = npdata.min(), npdata.max() * 1.05

    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    for r in range(rows):
        for c in range(cols):
            i = cols * r + c

            params, _, (dxg, _), _ = datas[i]
            sigma_cut = params[10]
            # if r == 0:
            #     axs[r, c].get_xaxis().set_visible(False)
            if c != 0 and c != (cols - 1):
                axs[r, c].get_yaxis().set_visible(False)
            if c == (cols - 1):
                axs[r, c].yaxis.set_label_position("right")
                axs[r, c].yaxis.tick_right()

            axs[r, c].set_ylabel("$g(r)$")
            if r == 1:
                axs[r, c].set_xlabel("r")

            axs[r, c].set_xlim(xmin, xmax)
            axs[r, c].set_ylim(ymin, ymax)

            y = npdata[i]
            x = np.arange(len(y)) * dxg
            axs[r, c].plot(x, y)

            text = f"{sigma_cut}"
            if i == 0:
                text = r"$2^{1/6}$"
            axs[r, c].set_title(r"$\sigma_c$=" + text, y=0.15, pad=0)
    plt.suptitle(f"Pair Correlation Function between x={xmin} and x={xmax} for different " + r"$\sigma_c$")
    plt.tight_layout(pad=1, w_pad=-0.25, h_pad=1)
    plt.show()


def plot_energy_discrepancy(energies):
    plt.figure()
    plt.plot((energies - energies[0]) / energies[0])
    plt.show()


def plot_energy(energies):
    plt.figure()
    plt.plot(energies)
    plt.show()


def plot_temp(params, velocities):
    N = params[0]
    par_every = params[3]
    x = np.arange(velocities.shape[0]) * par_every
    vsquared = np.linalg.norm(velocities, axis=2) ** 2
    temp = np.sum(vsquared, axis=1) / (3 * N)

    plt.figure()
    plt.plot(x, temp)
    plt.show()


def plot_vel_distribution(velocities):
    squared = np.linalg.norm(velocities, axis=1)
    squared = velocities[:, 0]
    x = np.linspace(0, 5, 1000)
    y = maxwell.pdf(x)
    plt.figure()
    plt.plot(x, y)
    plt.hist(squared, bins=15, density=True)
    plt.show()


def get_stats_nvt(datas):
    res = np.zeros((len(datas), 5))
    off = 1000
    for i, (params, stats, _, _) in enumerate(datas):
        N = params[0]
        en_k = stats[off:, 0]
        temps = stats[off:, 2]

        res[i, 0] = N
        res[i, 1] = 3 * N * 2 / 2  # Ek th = 3N T / 2
        res[i, 2] = 2 / (3 * N)  # th fluctuations
        res[i, 3] = en_k.mean()
        res[i, 4] = temps.var() / (temps.mean() ** 2)

    return res


def main():
    bpath = r""
    bname = "_ex10_TH2_NSIM8_M100000.bin"
    fname_stats = bpath + "stats" + bname
    fname_particles = bpath + "particles" + bname
    datas = read_binary_data(fname_stats, fname_particles)
    print("File Loaded")
    # res = get_stats_nvt(datas)
    # plot_pair_corr(datas)
    # # plot_vel_distribution(datas[0][-1][0, :, 3:6])
    # plot_temp(datas[0][0], datas[0][-1][:, :, 3:6])
    # plot_energy(energies)
    # plot_energy_discrepancy(energies)
    # animate_particles_3d(datas[0][0], datas[0][-1])


main()
