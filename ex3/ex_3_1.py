import numpy as np

np.random.seed(0xDEAF)


def crude_MC(n, f, x1, x2):
    points = np.random.uniform(x1, x2, n)
    y = f(points)
    return (x2 - x1) / n * np.sum(y)


def get_f(x):
    return np.sin(x)


def get_g(x):
    # b = 24/np.pi**3
    b = 0.45
    a = 2 * (1 - (b * np.pi ** 3) / 24) / np.pi
    return a + b * x ** 2


def get_g2(x):
    c = 384 / (48 * np.pi ** 2 - np.pi ** 4)
    return c * (x - (x ** 3) / 6)
    # return x * 8 / (np.pi ** 2)


def hit_and_miss_sampling(func, n, x1, x2, y1, y2):
    total_hits = np.array([], dtype=float)
    while len(total_hits) < n:
        points = np.random.uniform(low=[x1, y1], high=[x2, y2], size=(n, 2))
        hits = points[:, 1] < func(points[:, 0])
        total_hits = np.concatenate([total_hits, points[hits, 0]])

    return total_hits[:n]
    # nbins = int(np.sqrt(n / 16))
    # plt.hist(total_hits, density=True, bins=nbins)
    # plt.plot(np.linspace(x1, x2, 1000), func(np.linspace(x1, x2, 1000)))
    # plt.show()


def importance_sampling(rho, g, n, x1, x2):
    maxg = np.max(g(np.linspace(x1, x2, 10000)))
    g_points = hit_and_miss_sampling(g, n, x1, x2, 0, maxg * 1.05)
    h_points = rho(g_points) / g(g_points)
    return h_points


def importance_MC(rho, g, n, x1, x2):
    h_points = importance_sampling(rho, g, n, x1, x2)
    est_imp = np.sum(h_points) / n
    # dev_imp = np.abs(1 - est_imp)
    return est_imp


def run_find_under_threshold(experiment, dev, true_value, avg=100):
    n_tot = 0
    for i in range(avg):
        n = 1
        while True:
            est = experiment(n)
            dev_now = np.abs(true_value - est)
            if dev_now <= dev:
                n_tot += n
                break
            n += 1
    return n_tot / avg


def run_experiment(n=None, err_dev_perc=None):
    if err_dev_perc:
        avg = 1000
        n_imp = run_find_under_threshold(lambda n: importance_MC(get_f, get_g, n, 0, np.pi / 2), 0.01, 1, avg=avg)
        n_imp2 = run_find_under_threshold(lambda n: importance_MC(get_f, get_g2, n, 0, np.pi / 2), 0.01, 1, avg=avg)
        n_cs = run_find_under_threshold(lambda n: crude_MC(n, get_f, 0, np.pi / 2), 0.01, 1, avg=avg)
        print(f"Average iteration Crude MC: {n_cs}")
        print(f"Average iteration Importance Sampling (a+bx^2): {n_imp}")
        print(f"Average iteration Importance Sampling (Third degree Taylor in x=0): {n_imp2}")

    elif n:
        h_points1 = importance_sampling(get_f, get_g, n, 0, np.pi / 2)
        h_points2 = importance_sampling(get_f, get_g2, n, 0, np.pi / 2)
        est_imp = np.sum(h_points1) / n
        imp_es2 = np.sum(h_points2) / n
        crude_est = crude_MC(n, get_f, 0, np.pi / 2)

        print(f"Crude MC estimate: {crude_est}, err: {np.abs(1 - crude_est)}")
        print(f"Importance Sampling estimate (a+bx^2): {est_imp}, err: {np.abs(1 - est_imp)}")
        print(f"Importance Sampling estimate (Third degree Taylor in x=0): {imp_es2}, err: {np.abs(1 - imp_es2)}")


run_experiment(err_dev_perc=0.01)
run_experiment(n=10000000)
