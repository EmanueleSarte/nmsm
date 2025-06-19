import struct

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation


def read_binary_data(fname_stats):
    datas = []
    with (open(fname_stats, 'rb') as f):
        nsim = np.frombuffer(f.read(4), dtype=np.int32)[0]

        for k in range(nsim):
            N, niters, dt, cellside, ncellside, side, temp, eps, ncell = struct.unpack("=iiddidddi",
                                                                                       f.read(4 * 4 + 8 * 5))

            r1, r2, D1, D2, mu1, mu2, density, save_par = struct.unpack("=dddddddB", f.read(8 * 7 + 1))

            params = N, niters, dt, cellside, ncellside, side, temp, eps, ncell, r1, r2, D1, D2, mu1, mu2, density
            types = np.frombuffer(f.read(N * 4), dtype=np.int32)

            if save_par:
                out = np.frombuffer(f.read(2 * N * niters * 8)).reshape(niters, N, 2)
                # only1_flt = types == 1
                # only2_flt = types == 2
                #
                # y1 = (out[:, only1_flt, 0] - out[0, only1_flt, 0]) ** 2
                # y2 = (out[:, only2_flt, 0] - out[0, only2_flt, 0]) ** 2
                # out = [y1, y2]
            else:
                out = None

            datas.append((params, types, out))

    return datas


def animate_particles(data, types, side, skip=1, dt=0.2):
    num_frames = data.shape[0] // skip
    num_particles = data.shape[1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    # max_val = np.max(np.abs(data[:, :, 1:]))
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect('equal')

    typ1 = types == 1
    typ2 = types == 2

    scatter1 = ax.scatter(data[0, typ1, 0], data[0, typ1, 1], c="C0", s=31.25)
    scatter2 = ax.scatter(data[0, typ2, 0], data[0, typ2, 1], c="C1", s=25)
    text = ax.text(side*0.4, side * 0.95, '', fontsize=12)

    def update(frame):
        scatter1.set_offsets(data[frame * skip, typ1, :])
        scatter2.set_offsets(data[frame * skip, typ2, :])
        text.set_text(f"Time: {frame * skip * dt:.1f}")
        return scatter1, scatter2, text

    def progress(i, tot):
        if i % 100 == 0 and i:
            print(f"{i}/{tot}")

    # ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200/6)
    # writermp4 = animation.FFMpegWriter(fps=30)
    # ani.save("anim.mp4", writer=writermp4, progress_callback=progress, dpi=240)
    plt.show()


def deaccumulate_data_for_log(n, x, y):
    last = x[1] - x[0]
    delta = np.log10((len(x) + 1) / last) / n

    accx = None
    accy = None
    xi = last
    endi = 0
    started = False
    off = 0
    j = 0
    for i in range(n):
        xip1 = xi * (10 ** delta)
        xi = xip1

        starti = round(endi)
        endi = round(xip1)

        if (endi - starti > 1) and (not started):
            started = True
            off = starti
            accx = np.zeros(n - i)
            accy = np.zeros(n - i)

        if started:
            accx[j] = np.mean(x[starti:endi])
            accy[j] = np.mean(y[starti:endi])
            j += 1

    return np.concatenate([x[:off], accx]), np.concatenate([y[:off], accy])


def plot_msd_over_time(datas):
    dt = datas[0][0][2]
    niters = datas[0][0][1]
    time = np.arange(niters) * dt

    plt.figure(figsize=(10, 5))
    for i, (params, types, msd) in enumerate(datas):
        n = params[0]
        if n < 20:
            continue

        y1 = msd[0].mean(axis=1)
        y2 = msd[1].mean(axis=1)
        deaccx1, deaccy1 = deaccumulate_data_for_log(500, time, y1)
        deaccx2, deaccy2 = deaccumulate_data_for_log(500, time, y2)
        p = plt.plot(deaccx1, deaccy1, label=f"N={n}, type=A")
        plt.plot(deaccx2, deaccy2, label=f"N={n}, type=B", color=p[0].get_color(), ls="dashed")
        # p = plt.plot(time, y1, label=f"N={n}, type=A")
        # plt.plot(time, msd[1].mean(axis=1), label=f"N={n}, type=B", color=p[0].get_color(), ls="dashdot")

    plt.xlabel("Time (s)")
    plt.ylabel("Mean Square Displacement")
    plt.title(f"MSD of the x-position of the particles (dt={dt})")

    plt.axvline()
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def main():
    bname = r""
    fname = bname + "\\" + "stats_ex13_nsim6_M20000.bin"
    datas = read_binary_data(fname)
    print("Data Loaded")

    animate_particles(datas[5][2][:], datas[5][1], 100, skip=10)
    # plot_msd_over_time(datas)


main()
