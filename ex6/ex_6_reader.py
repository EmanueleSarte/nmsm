import struct

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation


def read_binary_data(filename):
    filename_stats = r"\\wsl.localhost\Ubuntu\home\emanuele\nmsm\stats" + filename
    with open(filename_stats, 'rb') as f:
        iters, n, c, maxc, maxtheta, h, ktot, save_pol, skip = struct.unpack(
            '=iiddddibi', f.read(4 + 4 + 8 + 8 + 8 + 8 + 4 + 4 + 1))
        betas = np.frombuffer(f.read(ktot * 8))
        observables = np.frombuffer(f.read(3 * 8 * ktot * iters)).reshape((iters, ktot, 3))
        try:
            swap_stats = np.frombuffer(f.read(4 * ktot * 4), dtype=np.int32).reshape((ktot, 4))
        except Exception as e:  # old format without these info
            swap_stats = None

    if save_pol:
        filename_poly = r"\\wsl.localhost\Ubuntu\home\emanuele\nmsm\poly" + filename
        try:
            with open(filename_poly, 'rb') as f:
                npoly = (iters // skip)
                polymers = np.frombuffer(f.read(n * 2 * 8 * ktot * npoly)).reshape((npoly, ktot, n, 2))
        except Exception as e:
            polymers = None
    else:
        polymers = None

    params = iters, n, c, maxc, maxtheta, h, ktot, skip, betas
    return params, observables, swap_stats, polymers


def plot_avg_polymer(params, observables, polymers, k):
    niters, N, c, maxc, maxtheta, h, ktot, skip, betas = params
    # energies = observables[:, :, 0]

    avg_x = np.mean(polymers[:, k, :, 0], axis=0)
    avg_y = np.mean(polymers[:, k, :, 1], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x_min = np.min(polymers[:, k, :, 0])
    x_max = np.max(polymers[:, k, :, 0])
    y_max = np.max(polymers[:, k, :, 1]) * 1.25

    ax.set_xlim(x_min - 1, x_max + 1)  # Add some padding
    ax.set_ylim(0, y_max + 1)
    ax.set_aspect('equal', adjustable='box')  # Ensure circles look like circles
    ax.set_title("Polymer Evolution")
    ax.hlines(y=h, xmin=x_min, xmax=x_max, colors="red", linestyle=':')
    ax.plot(avg_x, avg_y, linestyle='-', color='black', zorder=4)
    ax.plot(polymers[::10, k, -1, 0].flatten(), polymers[::10, k, -1, 1].flatten(), 'o',
            markersize=20, color="C0", alpha=0.02)

    beta = betas[k]
    ax.set_title(fr"$\beta={beta:.2f}$ - T={1 / beta:.2f} - No Chain Swapping - " + r"$n_{iters}$=" + f"{niters}")
    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Height")

    plt.tight_layout()
    plt.show()


def plot_polymer_instant(niter, params, observables, polymers, indexes):
    niters, N, c, maxc, maxtheta, h, ktot, skip, betas = params
    energies = observables[:, :, 0]

    nfig = len(indexes)
    fig, axs = plt.subplots(1, nfig, figsize=(nfig * 5 + 2, 5))

    try:
        axs[0]
    except Exception as e:
        axs = [axs]

    plot_dots = N < 50
    # plot_dots = True
    for j, ax in enumerate(axs):
        k = indexes[j]
        x_min = np.min(polymers[niter, k, :, 0])
        x_max = np.max(polymers[niter, k, :, 0])
        y_max = np.max(polymers[niter, k, :, 1]) * 1.25

        ax.set_xlim(x_min - 1, x_max + 1)  # Add some padding
        ax.set_ylim(0, y_max + 1)
        ax.set_aspect('equal', adjustable='box')  # Ensure circles look like circles
        ax.set_title("Polymer Evolution")
        ax.hlines(y=h, xmin=x_min, xmax=x_max, colors="red", linestyle=':')
        ax.plot(polymers[niter, k, :, 0], polymers[niter, k, :, 1], linestyle='-', color='black')

        if plot_dots:
            ax.plot(polymers[niter, k, :, 0], polymers[niter, k, :, 1], 'o', markersize=1, color="red", zorder=4)
            # ax.plot(polymers[:, k, :, 0].flatten(), polymers[:, k, :, 1].flatten(), 'o', markersize=20, color="C0", alpha=0.01, zorder=4)

        for i in range(N):
            circle = plt.Circle((polymers[niter, k, i, 0], polymers[niter, k, i, 1]),
                                radius=0.5, alpha=1, zorder=3)  # Initialize circles
            ax.add_patch(circle)

        beta = betas[k]
        ax.set_title(
            fr"$\beta={beta:.2f}$ - T={1 / beta:.2f} - Iter={niter * skip} - Energy: {energies[niter * skip, k]}")
        ax.set_xlabel("Distance from origin")
        ax.set_ylabel("Height")

    plt.tight_layout()
    plt.show()


def animate_polymers(params, observables, polymers, indexes):
    niters, N, c, maxc, maxtheta, h, ktot, skip, betas = params
    npoly = polymers.shape[0] // skip
    energies = observables[:, :, 0]

    nfig = len(indexes)
    fig, axs = plt.subplots(nfig, 1, figsize=(8, nfig * 2 + 2))

    try:
        axs[0]
    except Exception as e:
        axs = [axs]

    plot_dots = False
    lines = []
    dots = []
    texts = []
    circles = [[] for a in range(nfig)]
    for j, ax in enumerate(axs):
        k = indexes[j]
        x_min = np.min(polymers[:, k, :, 0])
        x_max = np.max(polymers[:, k, :, 0])
        y_max = np.max(polymers[:, k, :, 1]) * 0.5

        ax.set_xlim(x_min - 1, x_max + 1)  # Add some padding
        ax.set_ylim(0, y_max + 1)
        ax.set_aspect('equal', adjustable='box')  # Ensure circles look like circles
        # ax.set_title("Polymer Evolution")
        ax.hlines(y=h, xmin=x_min, xmax=x_max, colors="red", linestyle=':')
        line, = ax.plot([], [], linestyle='-', color='black')
        lines.append(line)

        if plot_dots:
            dot, = ax.plot([], [], 'o', markersize=1, color="red", zorder=4)
            dots.append(dot)

        for i in range(N):
            circle = plt.Circle((0, 0), radius=0.5, alpha=1, zorder=3)  # Initialize circles
            ax.add_patch(circle)
            circles[j].append(circle)

        text = ax.text(0, y_max * 0.9, '', horizontalalignment='center', verticalalignment='center', fontsize=12)
        texts.append(text)

    def progress(i, tot):
        if i % 100 == 0 and i:
            print(f"{i}/{tot}")

    def update(frame):
        for j, ax in enumerate(axs):
            k = indexes[j]
            for i, circle in enumerate(circles[j]):
                circle.center = (polymers[frame, k, i, 0], polymers[frame, k, i, 1])
            lines[j].set_data(polymers[frame, k, :, 0], polymers[frame, k, :, 1])
            texts[j].set_text(f"Beta={betas[k]}, Time: {frame * skip}, Energy: {energies[frame * skip, k]}")

            if plot_dots:
                dots[j].set_data(polymers[frame, k, :, 0], polymers[frame, k, :, 1])

        return *lines, *dots, *[c for cl in circles for c in cl], *texts

    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=npoly, interval=25, blit=True)
    writermp4 = animation.FFMpegWriter(fps=40)
    filename = "animation.mp4"
    ani.save(filename, writer=writermp4, progress_callback=progress, dpi=360)
    plt.show()


def calculate_observables(params, observables):
    iters, n, c, maxc, maxtheta, h, ktot, skip, betas = params

    energies = observables[:, :, 0]
    distances = observables[:, :, 1]
    heights = observables[:, :, 2]

    min_en = np.min(energies)
    plt.figure()
    for k in range(ktot):
        plt.hist(energies[500:, k], label=rf"$\beta$={betas[k]:.2f}, T={1 / betas[k]:.2f}",
                 bins=np.arange(min_en, 1, 1) + 0.5, alpha=0.5,
                 density=True)
    plt.legend()
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.title(fr"Energy Histograms - N={n - 1}" + r" - $n_{iters}=$ " + f"{iters} - With Chain Swap")
    plt.tight_layout()
    plt.show()


filename = ""
params, observables, swap_stats, polymers = read_binary_data(filename)
print("File Loaded")
calculate_observables(params, observables)
if polymers is not None:
    # plot_avg_polymer(params, observables, polymers, 9)
    animate_polymers(params, observables, polymers[:1000, ...], indexes=[0, 2, 4])
