import struct
import subprocess
import time

import numpy as np
from matplotlib import pyplot as plt, animation


def get_ising_data(nside: int, J: float, kB: float, T: float, nsteps: int, filename=None):
    if not filename:
        filename = "./ising_data.bin"
    call_cpp_ising(filename=filename, nx=nside, ny=nside, J=J, kB=kB, T=T, nsteps=nsteps)
    return read_binary_data(filename)


def call_cpp_ising(filename: str, nx: int, ny: int, J: float, kB: float, T: float, nsteps: int):
    cmd = f"./ex_5_ising filename={filename} nx={nx} ny={ny} J={J} kB={kB} T={T} nsteps={nsteps}"
    t1 = time.time()
    srun = subprocess.run(cmd, capture_output=True)
    print(f"Time to execute the file: {time.time() - t1: .3f} seconds")
    print(srun.stdout)
    if srun.returncode != 0:
        raise ValueError(f"Error while calling the cpp Ising file: {cmd}")


def read_binary_data(filename):
    with open(filename, 'rb') as f:
        iters, nx, ny, kB, T, J = struct.unpack('=iiiddd', f.read(4 + 4 + 4 + 8 + 8 + 8))

        energies = np.zeros(shape=iters)
        magnetizations = np.zeros(shape=iters, dtype=int)
        lattices = np.zeros(shape=(iters, nx, ny), dtype=np.int8)

        for i in range(iters):
            energy, magnet = struct.unpack('=di', f.read(4 + 8))
            lattice = np.frombuffer(f.read(nx * ny), dtype=np.int8).reshape((nx, ny))

            energies[i] = energy * J
            magnetizations[i] = magnet
            lattices[i] = lattice

    assert nx == ny
    params = iters, nx, ny, kB, T, J
    return params, energies, magnetizations, lattices


def animate_spins(energies, magnetzs, lattices):
    fig, ax = plt.subplots()
    im = ax.imshow(lattices[0], cmap='viridis', animated=True)

    def update(frame):
        i = frame * 1
        im.set_array(lattices[i])
        ax.set_title(f'Step: {i:04}, E: {energies[i]:.4f}, m: {magnetzs[i]:.4f}')
        # return im, title

    def progress(i, tot):
        if i % 100 == 0 and i:
            print(f"{i}/{tot}")

    # ani = animation.FuncAnimation(fig, update, frames=lattices.shape[0], interval=20, blit=False)
    ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False)
    writermp4 = animation.FFMpegWriter(fps=40)
    filename = "animation.mp4"
    ani.save(filename, writer=writermp4, progress_callback=progress, dpi=360)
    plt.show()


def plot_statistics(epochs, energies, magnetizs, cutoffs, nside, ratio_temp):
    en_eq_idx, mg_eq_idx = cutoffs

    plt.plot(epochs, energies, label="Energy per site")
    plt.plot(epochs, magnetizs, label="Magnetization per site")
    plt.vlines(x=epochs[en_eq_idx], ymin=min(energies.min(), magnetizs.min()),
               ymax=max(energies.max(), magnetizs.max()), colors="red", ls=":", label="Equilibrium")
    plt.title(f"Energy and Magnetization per site (L={nside}, T=${ratio_temp}T_c$)")

    plt.legend()
    plt.tight_layout()
    plt.show()


def get_observable(side, kb, T, energies_eq, magnetizs_eq):
    energy = np.mean(energies_eq) / side ** 2
    mspin = np.mean(magnetizs_eq) / side ** 2

    specific_heat = np.var(energies_eq) / (kb * T ** 2) / side ** 2
    magnetic_sus = np.var(magnetizs_eq) / (kb * T) / side ** 2

    return energy, mspin, specific_heat, magnetic_sus


def calculate_autocorr(points):
    autocorr = np.zeros(shape=len(points))
    tmax = len(points)

    for t in range(tmax):
        foo = 1 / (tmax - t)

        tmp1 = np.sum(points[0:tmax - t] * points[t: tmax])
        tmp2 = np.sum(points[0:tmax - t])
        tmp3 = np.sum(points[t:tmax])

        c0_t = foo * tmp1 - (foo ** 2) * tmp2 * tmp3

        autocorr[t] = c0_t

    tau0_int = np.sum(autocorr / autocorr[0])
    return tau0_int


def calculate_error_squared(tau0_int, points):
    foo = (1 + 2 * tau0_int) / (len(points) - 1)
    bar = np.sum((points - np.mean(points)) ** 2)
    return foo * bar


def find_equilibrium(points, sensitive):
    max_distance = len(points) // 4
    chunk = max_distance // 5

    if sensitive:
        m2ok = 0.1
    else:
        m2ok = 0.5

    for i in range(len(points) // chunk):
        start = chunk * i
        end = min(start + max_distance, len(points))
        m1, b1 = np.polyfit(np.arange(start, end), points[start:end], 1)
        m2, b2 = np.polyfit(np.arange(start, start + chunk), points[start:start + chunk], 1)
        if np.abs(m1) < 0.05 and np.abs(m2) < m2ok:
            return start


def run_analysis(params, energies, magnetizs):
    _, side, _, kB, T, J, ratio_temp = params
    N = side * side

    if ratio_temp <= 0.95:
        sensitive = True
    else:
        sensitive = False

    en_eq_idx = find_equilibrium(energies, sensitive=sensitive) or 0
    mg_eq_idx = find_equilibrium(magnetizs, sensitive=sensitive) or 0
    print(f"Equilibrium for energies: {en_eq_idx}, for magnetizations: {mg_eq_idx}, I'll use the biggest one!")
    cutoff = max(en_eq_idx, mg_eq_idx)

    plot_statistics(np.arange(len(energies)), energies / N, magnetizs / N, [en_eq_idx, mg_eq_idx],
                    side, ratio_temp)

    energies_eq = energies[cutoff:]
    magnetizs_eq = magnetizs[cutoff:]
    epochs = np.arange(cutoff, len(energies))

    obs = get_observable(side, kB, T, energies_eq, magnetizs_eq)
    obs_energy, obs_magnetz, obs_cv, obs_chi = obs
    print(f"Energy per site: {obs_energy:.4g}, Magn. per site: {obs_magnetz:.4g}, "
          f"Specific Heat: {obs_cv:.4g}, Magnetic Susceptibility: {obs_chi:.4g}")

    tau0_en = calculate_autocorr(energies_eq / N)
    tau0_mg = calculate_autocorr(magnetizs_eq / N)
    print(f"Integrated autocorrelation time for energy: {tau0_en:.3g}, for magn: {tau0_mg:.3g}")

    s2_en = calculate_error_squared(tau0_en, energies_eq / N)
    s2_mg = calculate_error_squared(tau0_mg, magnetizs_eq / N)
    print(f"The Error squared for energy is {s2_en:.4g}, for magnetization is: {s2_mg:.4g}")


def main():
    tc = 2 / np.log(1 + np.sqrt(2))
    for L in [50, 100, 200]:
        for ratio in [0.5, 0.97, 1.2]:
            nside = L
            nsteps = {0.5: 2000, 0.97: 16000, 1.2: 8000}[ratio]
            filename = f"ising_data_L{nside}_Tc{ratio}_nsteps{nsteps}.bin"
            print(f"Analyzing Ising with L={nside}, T={ratio}Tc, nsteps={nsteps}")
            params, energies, magnetz, lattices = get_ising_data(nside=nside, J=1, kB=1, T=tc * ratio, nsteps=nsteps,
                                                                 filename=filename)
            # params, energies, magnetz, lattices = read_binary_data(filename)
            params = params + (ratio,)

            run_analysis(params, energies, magnetz)

            animate_spins(energies / nside ** 2, magnetz / nside ** 2, lattices)


main()
