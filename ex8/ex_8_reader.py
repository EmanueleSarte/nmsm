import struct

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def read_binary_data(fname_stats, fname_particles=None):
    with open(fname_stats, 'rb') as f:
        N, niters, nbuf, save_par_every, eps, sigma, rho, dmax, sizex, sizey, sizez, utail, _ = struct.unpack(
            "=iiiiddddddddd", f.read(4 * 4 + 8 * 9))

        all_datas = np.zeros((niters, 2))

        nread = 0
        for i in range(int(np.ceil(niters / nbuf))):
            nleft = min(nbuf, niters - nread)
            energies = np.frombuffer(f.read(nleft * 8))
            pressures = np.frombuffer(f.read(nleft * 8))

            all_datas[nread:nread + nleft, 0] = energies
            all_datas[nread:nread + nleft, 1] = pressures

            nread += nleft

    all_particles = None
    if fname_particles:
        try:
            all_particles = np.zeros((niters // save_par_every, N, 3))
            with open(fname_particles, "rb") as f:
                for i in range(all_particles.shape[0]):
                    all_particles[i, :, 0] = np.frombuffer(f.read(N * 8))
                    all_particles[i, :, 1] = np.frombuffer(f.read(N * 8))
                    all_particles[i, :, 2] = np.frombuffer(f.read(N * 8))

        except Exception as e:
            pass

    params = N, niters, nbuf, save_par_every, eps, sigma, rho, dmax, sizex, sizey, sizez, utail
    return params, all_datas, all_particles


def read_multiple_files(fnames):
    datas = []
    for fname in fnames:
        data = read_binary_data(fname)
        datas.append(data)
    return datas


def animate_particles_3d(data):
    num_frames, num_particles, _ = data.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot(data[0, :, 0], data[0, :, 1], data[0, :, 2], marker='o', linewidth=0)

    def update(frame):
        # scatter._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        # scatter.set_data_3d(data[frame, :, :])
        line.set_data(data[frame, :, 0:2].reshape(2, -1))
        line.set_3d_properties(data[frame, :, 2])
        ax.set_title(f"Time Step: {frame}")
        return line,  # Important: return the artist for blitting

    # Set axis limits (you might need to adjust these based on your data)
    x_min, x_max = data[:, :, 0].min(), data[:, :, 0].max()
    y_min, y_max = data[:, :, 1].min(), data[:, :, 1].max()
    z_min, z_max = data[:, :, 2].min(), data[:, :, 2].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)  # blit=True for faster animation

    plt.show()


def plot_data(data):
    data = data[50:]
    plt.figure()
    plt.plot(np.arange(len(data)), data[:, 0])
    plt.show()
    plt.figure()
    plt.plot(np.arange(len(data)), data[:, 1])
    plt.show()


def plot_multiple_data(temp, real_data, *datas):
    plt.figure()
    plt.plot(real_data[:, 0], real_data[:, 1], label="Real Data", marker="o", markersize=3, linewidth=0.5)
    for i, data in enumerate(datas):
        N, mean_p, mean_p_err, rhos = data
        # plt.plot(rhos, mean_p, marker='o', markersize=3, linewidth=0.5,)
        plt.errorbar(rhos, mean_p, yerr=mean_p_err, capsize=3,  linewidth=0.5,  label=f"Simulated N={N}",
                     marker='o', markersize=3)
    plt.legend()
    plt.xlabel(r"density $\rho$")
    plt.ylabel(r"pressure P")
    niters = 10000
    plt.title(rf"Pressure vs Density, $T^*={temp}$, " + r"$n_{iters}$=" + f"{niters}")
    plt.tight_layout()
    plt.show()


def combine_multiple_files(datas):
    mean_p = np.zeros(len(datas))
    mean_p_err = np.zeros(len(datas))
    rhos = np.zeros(len(datas))

    off = 500
    for i, (params, data, _) in enumerate(datas):
        N, niters, _, _, eps, sigma, rho, dmax, sizex, sizey, sizez, utail = params
        pressures = data[off:, 1]
        mean_p[i] = np.mean(pressures)
        mean_p_err[i] = np.sqrt(np.var(data[50:, 1]) / len(pressures))
        rhos[i] = rho

    return mean_p, mean_p_err, rhos


def main():
    base = r"\\wsl.localhost\Ubuntu\home\emanuele\nmsm\ex8\data"
    temp = 0.9
    if temp == 0.9:
        ndata = 23
        real_fname = "./LJ_T/LJ_T09.dat"
        piece = "0.900000"
        niterss = [100, 200, 300]
    elif temp == 2.0:
        ndata = 38
        real_fname = "./LJ_T/LJ_T2.dat"
        piece = "2.000000"
        niterss = [300]

    with open(real_fname) as f:
        data_real = np.loadtxt(f)

    all_datas = []
    for k in niterss:
        nfiles = [f"{base}\\LJ_T{piece}_rho{i}_N{k}_M10000.bin" for i in range(ndata)]
        datas = read_multiple_files(nfiles)
        cdatas = combine_multiple_files(datas)
        cdatas = [k, *cdatas]
        all_datas.append(cdatas)
    plot_multiple_data(temp, data_real, *all_datas)

main()
