import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def apply_pbc(data, side):
    return data - np.floor(data / side) * side


def diff_pbc(data, side):
    side2 = side / 2
    ydiff = np.diff(data)
    return np.where(ydiff > side2, ydiff - side, np.where(ydiff < -side2, side - ydiff, ydiff))


# def get_distance_pbc(data, side):
#     data_pbc = data - np.round(data / side) * side
#     if len(data.shape) >= 2:
#         return np.linalg.norm(data_pbc, axis=-1)
#     else:
#         return data_pbc


class LangevinIntegrator:
    NO_POTENTIAL = 0
    HARMONIC_POT = 1

    def __init__(self, niters: int, rep: int, dt, side, gammas=None, temps=1, ks=1):

        self.niters = niters
        self.rep = rep
        self.dt = dt
        self.ndim = 2
        self.potential = self.NO_POTENTIAL
        self.k = ks

        self.gamma = gammas
        self.temp = temps

        self.nvals = self._assert_integrity(gammas, temps, ks)

        if not isinstance(ks, np.ndarray):
            self.k = np.array([ks])

        self.m = 1
        self.eps = 1
        self.sigma = 1
        self.side = side

    def run_simulation(self):
        states = np.empty((self.niters, self.rep, self.nvals, self.ndim))
        # displacements = np.empty((self.niters - 1, self.rep, self.nvals, self.ndim))
        # states[0, ...] = np.ones(shape=(self.rep, self.nvals, self.ndim)) * (self.side / 2)
        # states[0, ...] = np.ones(shape=(self.rep, self.nvals, self.ndim))
        states[0, ...] = np.zeros(shape=(self.rep, self.nvals, self.ndim))

        const_term = np.sqrt(2 * self.temp * self.dt / (self.m * self.gamma)).reshape(-1, 1)
        if isinstance(self.gamma, np.ndarray):
            const_term2 = (self.dt / self.m / self.gamma).reshape(-1, 1)
        else:
            const_term2 = self.dt / self.m / self.gamma

        for i in range(1, self.niters):
            noise = np.random.normal(size=states[0].shape)
            force_term = self.get_force(states[i - 1])
            states[i] = states[i - 1] + noise * const_term + force_term * const_term2
            # displacements[i - 1] = states[i] - states[i - 1]
            states[i] = self.apply_pbc(states[i])

        # return states, displacements
        return states

    def apply_pbc(self, state):
        return apply_pbc(state, self.side)

    # def get_distance_pbc(self, state):
    #     return get_distance_pbc(state, self.side)

    def get_force(self, state):
        if self.potential == self.NO_POTENTIAL:
            return np.zeros_like(state)

        elif self.potential == self.HARMONIC_POT:
            center = np.array([0, 0])
            distance = (state - center)
            distance = distance - np.round(distance / self.side) * self.side
            return -self.k.reshape(-1, 1) * distance

    def _assert_integrity(self, *args):
        arrays = [arg for arg in args if isinstance(arg, np.ndarray)]

        if len(arrays) == 0:
            return 1
        if len(arrays) > 1:
            raise ValueError("I can accept only one array ")
        return len(arrays[0])


def plot(data):
    # times = np.arange(len(data)) * 0.001

    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


def plot_displ_distrib(pos, displacements, side, temps, gammas, ks):
    nvals = pos.shape[-2]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if isinstance(temps, np.ndarray):
        value_text = "temperatures"
        end_text = rf", $K={ks}$, $\gamma={gammas}$"
        prefix = r"$T^* = $"
        values = temps
    if isinstance(gammas, np.ndarray):
        value_text = r"$\gamma$s"
        end_text = rf", $K={ks}$, $T^*={temps}$"
        prefix = r"$\gamma = $"
        values = gammas
    if isinstance(ks, np.ndarray):
        value_text = r"$K$s"
        end_text = rf", $\gamma={gammas}$, $T^*={temps}$"
        prefix = f"$K = $"
        values = ks

    nbins = round(np.sqrt(len(displacements)) / 4)
    stats = np.zeros(shape=(nvals, 2, 4))
    for k, ax in enumerate(axs):
        plt.suptitle(f"Distribution of displacements for different {value_text} - Harmonic Potential{end_text}")
        if k == 0:
            ax.set_title("x displacement")
        else:
            ax.set_title("y displacement")

        for i in range(nvals):
            y = displacements[:, i, k]
            stats[i, k, 0] = pos[:, i, k].mean()
            stats[i, k, 1] = pos[:, i, k].var()
            stats[i, k, 2] = (y ** 2).mean()
            stats[i, k, 3] = y.mean()
            print(stats[i, k])
            ax.hist(y, bins=nbins, density=True, color=f"C{i}", alpha=0.3, label=f"{prefix}{values[i]:.2f}")
            ax.hist(y, bins=nbins, density=True, histtype='step', edgecolor=f"C{i}", linewidth=1.5)
            if k == 0:
                ax.set_xlabel(r"$\Delta x$")
            else:
                ax.set_xlabel(r"$\Delta y$")

            ax.set_ylabel("Density")

        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_displac_distrib_over_time(data, N, values, dt):
    nvals = data.shape[2]
    ndata = data.shape[0]
    times = [100, 200, 500, 2000]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    rdata = np.empty((len(times), N, nvals, 2))
    for i, time in enumerate(times):
        yx = data[time, :, :, :]
        rdata[i, :, :, :] = (yx - np.round(yx / 20) * 20)

    for k, (ax, time) in enumerate(zip(axs, times)):

        timedata = rdata[k, :, :, 0]
        for i in range(nvals):
            y = timedata[:, i]
            nbins = round(np.sqrt(len(y)) / 3)
            ax.hist(y, bins=nbins, density=True, color=f"C{i}", alpha=0.3, label=fr"$K = {values[i]}$")
            ax.hist(y, bins=nbins, density=True, histtype='step', edgecolor=f"C{i}", linewidth=1.5)

        ax.set_xlabel("time")
        ax.set_title(f"Time = {time * dt: .2f} ({time} steps)")
        ax.set_xlim(rdata.min(), rdata.max())
        ax.legend()
        if (k % 2) == 0:
            ax.set_ylabel("Mean Squared X-Disp")

    plt.suptitle("Mean Squared X-Displacement with PBC, " + rf"$T^*=1$, $\gamma\tau=1$, $V=-Kx^2$, avg of N={N} particles")

    plt.tight_layout(pad=1, w_pad=0.25, h_pad=1)
    plt.show()


def plot_msd_over_time(data, N, values, dt):
    nvals = data.shape[2]
    ndata = data.shape[0]

    plt.figure(figsize=(10, 4))

    maxy = 0
    for i in range(nvals):
        time = np.arange(ndata) * dt
        yx = data[:, :, i, 0]
        # dist_no_squared = (yx - np.round(yx / 20) * 20)
        # print(f"{values[i]:.6f} {dist_no_squared.mean():.6f} {dist_no_squared.var():.6f}")
        msdx = ((yx - np.round(yx / 20) * 20) ** 2).mean(axis=-1)
        # msdx = ((yx - np.round(yx / 20) * 20)).mean(axis=-1)

        plt.plot(time, msdx, color=f"C{i}", label=rf"$K = {values[i]}$")
        # plt.plot(time, 2 / values[i] * time, color=f"C{i}", ls=":")

        maxy = max(msdx.max(), maxy)

    plt.title(
        "Mean Squared X-Displacement with PBC, " + rf"$T^*=1$, $\gamma\tau=1$, $V=-Kx^2$, avg of N={N} particles, $\Delta t={dt}$")
    # plt.ylim(0, maxy * 1.1)
    plt.xlabel("time")
    plt.ylabel("Mean Squared X-Disp")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def animate_particles(data, side, ks):
    num_frames = data.shape[0]
    num_particles = data.shape[1]

    if not isinstance(ks, np.ndarray):
        ks = [ks]

    fig, ax = plt.subplots()
    # max_val = np.max(np.abs(data[:, :, 1:]))
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect('equal')

    colors = [f"C{i}" for i in range(num_particles)]
    scatter = ax.scatter(data[0, :, 0], data[0, :, 1], c=colors, s=70)
    tails = [ax.plot([], [], color=colors[i], lw=2, zorder=-1)[0] for i in range(num_particles)]

    delta = 0.02
    xgrid = np.arange(0, side, delta)
    ygrid = np.arange(0, side, delta)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid)
    z = np.sqrt((xmesh - side / 2) ** 2 + (ymesh - side / 2) ** 2) * ks[0]
    z = 1 - (z / (z.max()))
    ax.imshow(z, cmap='binary', interpolation='nearest', extent=[0, side, 0, side], vmin=0, vmax=1.5)

    def update(frame):
        scatter.set_offsets(data[frame, :, :])
        start = max(0, frame - 20)
        for i, tail in enumerate(tails):
            tail.set_xdata(data[start:frame, i, 0])
            tail.set_ydata(data[start:frame, i, 1])
        return scatter, *tails

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=30)
    plt.show()


def main():
    dt = 0.01
    side = 20
    ks = 1
    gammas = 1
    temps = 1

    # gammas = np.array([0.1, 0.5, 1, 2, 10])
    # temps = np.linspace(0.1, 2, 5)
    ks = np.array([0.1, 0.2, 0.5, 1, 10])

    rep = 5000
    lang_od_temps = LangevinIntegrator(niters=3000, rep=rep, side=side, ks=ks, dt=dt, gammas=gammas, temps=temps)
    lang_od_temps.potential = LangevinIntegrator.HARMONIC_POT
    states = lang_od_temps.run_simulation()

    plot_msd_over_time(states, rep, ks, dt)
    plot_displac_distrib_over_time(states, rep, ks, dt)

    # animate_particles(states[:, 0, :, :], side, ks)


main()
