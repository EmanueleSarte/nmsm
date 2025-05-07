import struct

import numpy as np
from matplotlib import pyplot as plt


def read_multiple_files(filenames):
    data = []
    for filename in filenames:
        out = read_binary_data(filename)
        data.append(out)
    return data


def read_binary_data(filename):
    with open(filename, 'rb') as f:
        iters, n, c, maxc, maxtheta, h, ktot, save_pol, skip = struct.unpack(
            '=iiddddibi', f.read(4 + 4 + 8 + 8 + 8 + 8 + 4 + 4 + 1))
        betas = np.frombuffer(f.read(ktot * 8))
        observables = np.frombuffer(f.read(3 * 8 * ktot * iters)).reshape((iters, ktot, 3))
        swap_stats = np.frombuffer(f.read(4 * ktot * 4), dtype=np.int32).reshape((ktot, 4))
        polymers = None

    params = iters, n, c, maxc, maxtheta, h, ktot, skip, betas
    return params, observables, swap_stats, polymers


def analyze_mean_height(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.75)

    for i, data in enumerate(datas):
        iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]
        observables = data[1]

        heights = observables[:, :, 2]
        mean_hei = np.mean(heights, axis=0)
        var_hei = np.var(heights, axis=0)
        p, = ax.plot(betas, mean_hei, marker='o', label=f"N={n - 1}, iters={iters}", color=f"C{i}", markersize=5)
        # ax.errorbar(betas, mean_hei, yerr=np.sqrt(var_hei / iters), capsize=3, ecolor=f"C{i}")

    plt.xticks(np.arange(0.5, np.max(betas) + 0.25, 0.25))
    plt.ylabel("Heights")
    plt.xlabel(r"$\beta$s")
    plt.title("Mean Heights")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_mean_energies(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.75)

    for i, data in enumerate(datas):
        iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]
        observables = data[1]

        energies = observables[:, :, 0]
        mean_en = np.mean(energies, axis=0)
        var_en = np.var(energies, axis=0)
        p, = ax.plot(betas, mean_en, marker='o', label=f"N={n - 1}, iters={iters}", color=f"C{i}", markersize=5)
        p, = ax.plot(betas, var_en, marker='o', label=f"N={n - 1}, iters={iters}", color=f"C{i}", markersize=5)

        # ax.errorbar(betas, mean_en, yerr=np.sqrt(var_en / iters), capsize=3, ecolor=f"C{i}")

    plt.xticks(np.arange(0.5, np.max(betas) + 0.25, 0.25))
    plt.ylabel("Energies")
    plt.xlabel(r"$\beta$s")
    plt.title("Mean Energies")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_mean_distance(datas):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.75)

    for i, data in enumerate(datas):
        iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]
        observables = data[1]

        distances = np.sqrt(observables[:, :, 1])
        mean_dist = np.mean(distances, axis=0)
        var_dist = np.var(distances, axis=0)

        ax.plot(betas, mean_dist ** 2 / n ** 2, marker='o', label=f"N={n - 1}, iters={iters}", color=f"C{i}",
                markersize=5)
        # ax.errorbar(betas, mean_dist, yerr=np.sqrt(var_dist / iters), capsize=3, ecolor=f"C{i}")

    plt.xticks(np.arange(0.5, np.max(betas) + 0.25, 0.25))
    plt.ylabel("Distance")
    plt.xlabel(r"$\beta$s")
    plt.title("Mean Distances Normalized")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def no_exp_format(value):
    a = f"{value:.1g}"
    if "e+" in a:
        exp = 10 ** int(a[-2:])
        return str(round(value / exp) * exp)

    return a


def show_errors(datas):
    buf_errors = None
    buf_data = None
    for i, data in enumerate(datas):
        iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]

        nd = len(datas)
        if buf_errors is None:
            buf_errors = np.zeros(shape=(ktot, 3 * nd))
            buf_data = np.zeros(shape=(ktot, 3 * nd))

        for obs in range(3):
            observable = data[1][:, :, obs]
            if obs == 1:
                observable = np.sqrt(observable)
            mean_obs = np.mean(observable, axis=0)
            var_obs = np.var(observable, axis=0)

            buf_data[:, nd * obs + i] = mean_obs
            buf_errors[:, nd * obs + i] = np.sqrt(var_obs / iters)

    # print([f"{a:.1g}" for a in buf_errors[2]])
    nef = no_exp_format
    # print([nef(a) for a in buf_errors[0]])
    print([f"{a}+-{b:.1g}" for a, b in zip(buf_data[0], buf_errors[0])])


def show_swap_stats(data):
    iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]
    swap = np.array(data[2], dtype=float)
    swap[:, [0, 1, 2, 3]] = swap[:, [1, 2, 3, 0]]

    swap_perc = np.zeros(shape=(ktot - 1, 4))
    swap_perc[:, 0:3] = swap[:-1, 0:3] / swap[:-1, 3].reshape(-1, 1)
    swap_perc[:, 3] = np.sum(swap_perc[:, 0:3], axis=1)

    for k in range(ktot - 1):
        print([f"{a:.2f}" for a in swap_perc[k, :]])


def acf(x, length=20):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])


def analyze_autocorr(data):
    plt.figure()
    iters, n, c, maxc, maxtheta, h, ktot, skip, betas = data[0]

    for k in [0,-1]:
        points = data[1][:, k, 0]
        # autocorr = np.zeros(shape=len(points))
        tmax = len(points)
        autocorr = acf(points, 10000)

        # for t in range(tmax):
        #     foo = 1 / (tmax - t)
        #
        #     tmp1 = np.sum(points[0:tmax - t] * points[t: tmax])
        #     tmp2 = np.sum(points[0:tmax - t])
        #     tmp3 = np.sum(points[t:tmax])
        #
        #     c0_t = foo * tmp1 - (foo ** 2) * tmp2 * tmp3
        #
        #     autocorr[t] = c0_t

        tau0_int = np.sum(autocorr / autocorr[0])
        plt.plot(autocorr, label=rf"$\beta$={betas[k]}")

    plt.show()


def main():
    base = r""
    filenames = [
        base + r"\stats_buono_N25_M20000.bin",
        base + r"\stats_buono_N50_M20000.bin",
        base + r"\stats_buono_N75_M40000.bin",
        base + r"\stats_buono_N100_M40000.bin",
        base + r"\stats_buono_N150_M50000.bin",
        base + r"\stats_buono_N200_M80000.bin",

    ]
    datas = read_multiple_files(filenames)
    analyze_mean_energies(datas)
    analyze_mean_distance(datas)
    analyze_mean_height(datas)
    show_errors(datas)
    analyze_autocorr(datas[-1])
    # show_swap_stats(datas[-1])


main()
