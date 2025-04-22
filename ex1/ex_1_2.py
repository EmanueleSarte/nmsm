import numpy as np
import matplotlib.pyplot as plt


def get_pdf1(x):
    return 2 * x * np.exp(-x ** 2)


def get_inv_cumul1(x):
    return np.sqrt(np.log(1 / (1 - x)))


def get_pdf2(x):
    return 5 * (x ** 4) / 243


def get_inv_cumul2(x):
    return np.float_power(243 * x, 0.2)


def create_plot(ax, hist_values, pdf, title):
    linspace = np.linspace(min(hist_values), max(hist_values), 1000)

    nbins = int(np.sqrt(len(hist_values) / 4))
    ax.hist(hist_values, bins=nbins, density=True, label="Sampled")
    ax.plot(linspace, pdf(linspace), label="PDF")
    ax.set_xlabel('x')
    ax.set_ylabel('pdf')
    ax.set_title(title)
    ax.legend()


def main():
    npoints = 100000
    points1 = np.random.uniform(1e-8, 1 - 1e-8, npoints)
    values1 = get_inv_cumul1(points1)

    points2 = np.random.uniform(0, 1, npoints)
    values2 = get_inv_cumul2(points2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    create_plot(ax1, values1, get_pdf1, r'$\rho(x) = 2xe^{-x^2}$' + f' ({npoints} points)')
    create_plot(ax2, values2, get_pdf2, r'$\frac{5}{243}x^4$' + f' ({npoints} points)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()