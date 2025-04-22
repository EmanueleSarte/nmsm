import numpy as np
import matplotlib.pyplot as plt


def is_inside_ellipsoid(points, a, b, c):
    return (points[:, 0] / a) ** 2 + (points[:, 1] / b) ** 2 + (points[:, 2] / c) ** 2 <= 1


def estimate_volume(n, a, b, c):
    vol_box = 8 * a * b * c
    points = np.random.uniform(low=[-a, -b, -c], high=[a, b, c], size=(n, 3))
    inside = is_inside_ellipsoid(points, a, b, c)
    hits = np.sum(inside)
    return vol_box * hits / n


def run_experiment(n_estimates, a, b, c):
    exact_vol = (4 / 3) * np.pi * a * b * c
    data = np.zeros(shape=(n_estimates, 4))

    for i, n in enumerate(np.logspace(2, 7, n_estimates, dtype=int)):
        est_vol = estimate_volume(n, a, b, c)
        dev = abs(est_vol - exact_vol)
        data[i, 0] = n
        data[i, 1] = exact_vol
        data[i, 2] = est_vol
        data[i, 3] = dev

    return data

def run_averaged_experiments(n_estimates, a, b, c):

    n_total = 20
    totaldata = np.zeros(shape=(n_estimates, 4))
    for i in range(n_total):
        data = run_experiment(n_estimates, a, b, c)

        if i == 0:
            totaldata[:, 0] = data[:, 0]
            totaldata[:, 1] = data[:, 1]

        totaldata[:, 2:] += data[:, 2:]

    totaldata[:, 2:] /= n_total
    return totaldata



data1 = run_averaged_experiments(12, 3, 2, 2)
data2 = run_averaged_experiments(12, 3, 1, 1)

plt.figure(figsize=(10, 6))
plt.plot(data1[:, 0], data1[:, 3], label='Ellipsoid (3,2,2)', marker='o')
plt.plot(data2[:, 0], data2[:, 3], label='Ellipsoid (3,1,1)', marker='s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of samples (log scale)')
plt.ylabel('Deviation from true volume (log scale)')
plt.title('Deviation vs Number of Samples')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

print("By using the bounding box [-a, -b, -c]x[a, b, c] we get:")
print(f"Exact volume for ellipsoid (3,2,2): {data1[0, 1]:.4f}, best estimate: {data1[-1, 2]}")
print(f"Exact volume for ellipsoid (3,1,1): {data2[0, 1]:.4f}, best estimate: {data2[-1, 2]}")
