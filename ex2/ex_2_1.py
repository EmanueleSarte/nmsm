import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

K_1_4_1 = 0.430739774448585

C = 2.5


def get_pdf(x):
    return np.sqrt(2) / (np.e * K_1_4_1) * np.exp(-8 * ((x ** 2) / 2 + (x ** 4) / 4))


def get_easy_pdf(x):
    return norm.pdf(x, loc=0, scale=1 / np.sqrt(2))


def sample_easy_pdf(n):
    return np.random.normal(loc=0, scale=1 / np.sqrt(2), size=n)


def run_experiment(n):
    points = sample_easy_pdf(n)
    pdf_y = get_pdf(points)
    easy_pdf_y = get_easy_pdf(points)
    ratio = pdf_y / (easy_pdf_y * C)

    unif_points = np.random.uniform(size=n)
    points_good = points[unif_points < ratio]
    return points_good


points_sampled = run_experiment(100000)

x_plot = np.linspace(min(points_sampled), max(points_sampled), 1000)
y_plot = get_pdf(x_plot)

plt.hist(points_sampled, density=True, bins=int(np.sqrt(len(points_sampled) / 4)), label="Sampled")
plt.plot(x_plot, y_plot, label="Real PDF")
plt.legend()
plt.title(r"Sampling $\frac{\sqrt{2}}{e K_{1/4}(1)}e^{-8(x^2/2+x^4/4)}$")
plt.show()
