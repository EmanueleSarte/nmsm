import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6.4, 6.4))
t = np.linspace(0, 3, 40)
g = -9.81
v0 = 12
z = g * t ** 2 / 2 + v0 * t

v02 = 5
z2 = g * t ** 2 / 2 + v02 * t

scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    x = t[:frame]
    y = z[:frame]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)


def progress(i, tot):
    if i % 100 == 0 and i:
        print(f"{i}/{tot}")


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
writermp4 = animation.FFMpegWriter(fps=40)
filename = "animation.mp4"
ani.save(filename, writer=writermp4, progress_callback=progress, dpi=300)
plt.show()
