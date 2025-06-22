import pickle
import struct

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt, animation
from scipy.optimize import curve_fit
from scipy.stats import linregress


def read_binary_data(fname_stats):
    datas = []
    with (open(fname_stats, 'rb') as f):
        nsim = np.frombuffer(f.read(4), dtype=np.int32)[0]

        for k in range(nsim):
            N, niters, kB, temp, J, save_spins = struct.unpack("=iidddB", f.read(4 * 2 + 8 * 3 + 1))
            params = N, niters, kB, temp, J, save_spins

            spins = None
            if save_spins:
                spins = np.frombuffer(f.read(N * N * niters * 1), dtype=np.int8).reshape(niters, N, N)

            stats = np.empty(shape=(niters, 2))
            stats[:, 0] = np.frombuffer(f.read(niters * 8))
            stats[:, 1] = np.frombuffer(f.read(niters * 8))

            stats = stats.astype(np.int32)
            datas.append((params, stats, spins))
    return datas


def plot_observables(n, n_iters, temps, energies):
    for i in range(1, len(temps) - 1):

        plt.figure()
        for j in range(i - 1, i + 2):
            temp = temps[j]
            beta = 1 / temp
            data = energies[j]
            delta = data.max() - data.min()
            nbins = round(np.log(delta) * 5)
            plt.hist(data, bins=nbins, label=f"T={temp:.3f}, $\\beta$={beta:.3f}", alpha=0.5, density=True)

        plt.title(f"Energy distributions, L={n}, n_iters={n_iters}")
        plt.xlabel("Energy")
        plt.legend()
        plt.show(block=True)


def calculate_cbetas(x_betas, zetas, betas, n, energies):
    M = len(x_betas)
    all_energies = np.concatenate(energies)
    unique, counts = np.unique(all_energies, return_counts=True)
    niters = len(all_energies) // M

    zeta_betas = np.zeros(M)
    en_betas = np.zeros(M)
    en2_betas = np.zeros(M)
    for k, xbeta in enumerate(x_betas):
        tot_zb = 0
        tot_en = 0
        tot_en2 = 0

        var = niters * (1 / zetas)
        deltabeta = xbeta - betas
        for i, en in enumerate(unique):
            argexp = deltabeta * en
            den = np.sum(var * np.exp(argexp))
            foo = counts[i] / den
            tot_zb += foo
            tot_en += en * foo
            tot_en2 += (en ** 2) * foo

        zeta_betas[k] = tot_zb
        en_betas[k] = tot_en
        en2_betas[k] = tot_en2

    en_betas = en_betas / zeta_betas
    en2_betas = en2_betas / zeta_betas

    c_betas = (en2_betas - en_betas ** 2) / (n ** 2) * (x_betas ** 2)
    real_c_betas = np.var(energies, axis=1) / (n ** 2) * (betas ** 2)

    return real_c_betas, c_betas


def calculate_Zk(betas, energies):
    M = len(betas)
    all_energies = np.concatenate(energies)
    unique, counts = np.unique(all_energies, return_counts=True)
    niters = len(all_energies) // M
    zetas = np.ones(M)
    new_zetas = np.zeros(M)
    m = 0
    while True:
        for k in range(M):
            tot = 0
            var = niters * (1 / zetas)
            deltabeta = betas[k] - betas
            for i, en in enumerate(unique):
                den = np.sum(var * np.exp(deltabeta * en))
                tot += counts[i] / den

            new_zetas[k] = tot

        delta2 = np.sum(((new_zetas - zetas) / new_zetas) ** 2)
        if delta2 < 1e-14:
            print("converged", m)
            break

        if m > 200:
            print("not converged")
            break
        m += 1

        factor = (new_zetas.min() * new_zetas.max()) ** (-0.5)
        new_zetas = new_zetas * factor

        zetas = new_zetas.copy()

    print(zetas)
    return zetas
    # zetas = np.ones(M)
    # new_zetas = np.zeros(M)
    # m = 0
    # while True:
    #     q = 1
    #     for k in range(M):
    #         tot = 0
    #         for i, en in enumerate(unique):
    #             nu = counts[i]
    #             l_kq = np.log(niters) - np.log(zetas[q]) + (betas[k] - betas[q]) * en
    #             part1 = np.exp(l_kq)
    #             part2 = 0
    #             for j in range(M):
    #                 l_kj = np.log(niters) - np.log(zetas[j]) + (betas[k] - betas[j]) * en
    #                 part2 += np.exp(l_kj - l_kq)
    #
    #             den = part1 * part2
    #             tot += nu / den
    #
    #         new_zetas[k] = tot
    #
    #     delta2 = np.sum(((new_zetas - zetas) / new_zetas) ** 2)
    #     if (delta2 < 1e-14) or m > 500:
    #         break
    #     m += 1
    # zetas = new_zetas.copy()


def plot_temp_vs_cbeta(ax, xtemps, cbetas, temps, creal):
    ax.plot(xtemps, cbetas)
    ax.scatter(temps, creal)


def plot_grid(spins, energy, magnet):
    fig, ax = plt.subplots()
    im = ax.imshow(spins, cmap='viridis', animated=True)
    plt.title(f"N={spins.shape[0]}, Energy={energy}, M={magnet}")
    plt.tight_layout()
    plt.show()


def fit_function(x, phi):
    return x ** (-phi)


# def perform_fit_analysis(n_peaks, beta_peaks):
#     x, y = n_peaks, beta_peaks
#     plt.plot(y)
#     plt.scatter(np.arange(len(y)), y)
#     plt.show()
#
#     popt, pcov, info_dict, mesg, ier = curve_fit(fit_function, x, y, bounds=(-5, 5), full_output=True)
#
#     phi = popt[0]
#     # phi = 0.1
#     plt.plot(x, fit_function(x, phi), 'r-', label=f'fit: phi={phi},')
#     plt.scatter(x, fit_function(x, phi))
#     plt.legend()
#     plt.show()


def perform_fit_analysis2(n_peaks, beta_peaks):
    y = beta_peaks

    phis = np.linspace(0.001, 2, 200)
    data = np.empty(shape=(len(phis), 4))
    for i, phi in enumerate(phis):
        x = n_peaks ** (-phi)
        slope, intercept, r, p, se = linregress(x, y)

        res = (y - (slope * x + intercept)) ** 2
        data[i, 0] = phi
        data[i, 1] = np.sum(res)
        data[i, 2] = slope
        data[i, 3] = intercept

    iphi = np.argmin(data[:, 1])
    phi, res, slope, intercept = data[iphi]
    print(f"phi={phi}, slope={slope}, intercept={intercept}, res={res}")
    print(f"Estimated Tc = {1/intercept}, true: {2 / np.log(1 + np.sqrt(2))}")
    x = n_peaks ** (-phi)
    xlin = np.linspace(x.min(), x.max(), 100)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(xlin, slope * xlin + intercept)
    ax1.scatter(x, y, color="C1")

    ax2 = ax1.twiny()
    ax2.scatter(x, y, color="C1", s=0)
    ax2.set_xlabel('Original L')
    # ax2.tick_params(axis='x')
    custom_tick_positions = x
    custom_tick_labels = [f"{n:.0f}" for n in n_peaks]
    ax2.set_xticks(custom_tick_positions)
    ax2.set_xticklabels(custom_tick_labels)

    ax1.set_xlabel("$N^{-\\phi}$")
    ax1.set_ylabel("$\\beta_{peak}$")
    plt.title(f"Best Line through points: phi={phi:.3f}, m={slope:.3f}, q={intercept:.3f}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    M = 200000
    # bname = r""
    # fnames = [
    #     bname + f"stats_ex_12_N20_M{M}.bin",
    #     bname + f"stats_ex_12_N30_M{M}.bin",
    #     bname + f"stats_ex_12_N40_M{M}.bin",
    #     bname + f"stats_ex_12_N50_M{M}.bin",
    #     bname + f"stats_ex_12_N60_M{M}.bin",
    #     bname + f"stats_ex_12_N70_M{M}.bin",
    #     bname + f"stats_ex_12_N80_M{M}.bin",
    #     bname + f"stats_ex_12_N90_M{M}.bin",
    #     bname + f"stats_ex_12_N100_M{M}.bin",
    #     bname + f"stats_ex_12_N110_M{M}.bin"
    # ]
    # nsim = len(fnames)
    # data_peaks = np.zeros(shape=(nsim, 2))
    # data_to_save = [None] * nsim
    # for i, fname in enumerate(fnames):
    #     datas = read_binary_data(fname)
    #     N = datas[0][0][0]  # N = the side of the grid
    #     n_iters = datas[0][0][1]
    #     temps = np.array([par[3] for par, _, _ in datas])
    #     betas = 1 / temps
    #
    #     discard = 10000
    #     energies = [stats[discard:, 0] for _, stats, _ in datas]
    #
    #     # plot_observables(N, n_iters, temps, energies)
    #     zks = calculate_Zk(betas, energies)
    #
    #     xtemps = np.linspace(temps.min() - 0.02, temps.max() + 0.02, 150)
    #     creal, ccalc = calculate_cbetas(1 / xtemps, zks, betas, N, energies)
    #
    #     data_to_save[i] = ([temps, creal, ccalc, xtemps])
    #
    #     ipeak = np.argmax(ccalc)
    #     tpeak = xtemps[ipeak]
    #
    #     data_peaks[i, 0] = N
    #     data_peaks[i, 1] = tpeak
    #
    # with open(f"data_ex_12_M{M}.bin", "wb") as f:
    #     pickle.dump([data_peaks, data_to_save], f)

    with open(f"data_ex_12_M{M}.bin", "rb") as f:
        data_peaks, data_to_save = pickle.load(f)

    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    axs = axs.flatten()

    for i, (temps, creal, ccalc, xtemps) in enumerate(data_to_save):

        axs[i].plot(xtemps, ccalc, color="C0", label="$C(\\beta)$ (MHM)")
        axs[i].scatter(temps, creal, color="C1", label="$C(\\beta^*)$", s=10)

        axs[i].set_ylabel("$C^{sp}$")

        r = i // 5
        c = i % 5
        if c != 0 and c != (5 - 1):
            axs[i].get_yaxis().set_visible(False)
        if c == (5 - 1):
            axs[i].yaxis.set_label_position("right")
            axs[i].yaxis.tick_right()

        if r == 1:
            axs[i].set_xlabel("$\\beta$")
        if i == 0:
            axs[i].legend(fontsize='small')
        if r == 0:
            axs[i].set_title(f"L={data_peaks[i, 0]:.0f}", y=0.35, pad=0)
        else:
            axs[i].set_title(f"L={data_peaks[i, 0]:.0f}", y=0.15, pad=0)

        tc = 2 / np.log(1 + np.sqrt(2))
        axs[i].axvline(x=tc, color="black", lw=1)

    plt.suptitle("C($\\beta$) for different L")
    plt.tight_layout(pad=1, w_pad=-0.25, h_pad=1)
    plt.show()

    n_peaks = data_peaks[:, 0]
    beta_peaks = 1 / data_peaks[:, 1]
    perform_fit_analysis2(n_peaks, beta_peaks)

main()
