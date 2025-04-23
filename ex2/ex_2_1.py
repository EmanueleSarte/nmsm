import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

K_1_4_1 = 0.430739774448585

C = 1.1
np.random.seed(0xCAFFE)

def get_pdf(x):
    return np.sqrt(2) / (np.e * K_1_4_1) * np.exp(-8 * ((x ** 2) / 2 + (x ** 4) / 4))


def get_easy_pdf(x):
    return norm.pdf(x, loc=0, scale=0.35)


def sample_easy_pdf(n):
    return np.random.normal(loc=0, scale=0.35, size=n)


def run_experiment(n):
    points = sample_easy_pdf(n)
    pdf_y = get_pdf(points)
    easy_pdf_y = get_easy_pdf(points)
    ratio = pdf_y / (easy_pdf_y * C)

    unif_points = np.random.uniform(size=n)
    points_good = points[unif_points < ratio]
    return points_good

def show_plot_pdf_vs_easy():
    plt.figure()
    x = np.linspace(-1.25, 1.25, 10000)
    y = get_pdf(x)
    yeasy = get_easy_pdf(x)
    plt.plot(x, y, label="f(x)")
    plt.plot(x, yeasy * C, label="c*g(x)")
    plt.legend()
    plt.title("Comparison between the real PDF ( $f(x)$ ) and the 'easy' one ( $g(x)$ )")
    plt.tight_layout()
    plt.savefig("ex_2_1_comparison.svg")
    plt.show()

show_plot_pdf_vs_easy()

N = 100000
points_sampled = run_experiment(N)

print(f"Efficiency: {len(points_sampled) / N * 100: 0.1f}%")

x_plot = np.linspace(min(points_sampled), max(points_sampled), 1000)
y_plot = get_pdf(x_plot)

plt.hist(points_sampled, density=True, bins=int(np.sqrt(len(points_sampled) / 8)), label="Sampled")
plt.plot(x_plot, y_plot, label="Original PDF")
plt.legend()
plt.title(r"Sampling $\frac{\sqrt{2}}{e K_{1/4}(1)}e^{-8(x^2/2+x^4/4)}$" + f"  (N = {N})")
plt.tight_layout()
plt.savefig("ex_2_1_sampling.svg")
plt.show()
