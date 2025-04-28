import numpy as np
from scipy.stats import norm


np.random.seed(0xF00D)


def get_f(x):
    return np.exp(-((x-3)**2)/2) + np.exp(-((x-6)**2)/2)

def get_rho(x):
    return norm.pdf(x, loc=0, scale=1)

def regular_MC(f, n):
    points = np.random.normal(loc=0, scale=1, size=n)
    h = f(points)
    estimate = np.sum(h) / n
    return estimate

def importance_sampling(f, rho, n):
    points = np.random.uniform(low=-8, high=-1, size=n)
    h_points = f(points) * rho(points) / (1/7)
    estimate = np.sum(h_points) / n
    return estimate

def run_experiment():
    est_MC = regular_MC(get_f, 1000)
    est_imp = importance_sampling(get_f, get_rho,1000)

    true_val = (np.exp(-9/4) + np.exp(-9)) / np.sqrt(2)
    true_val_interval = 0.0000151648
    print(f"Estimated value with regular MC: {est_MC:.5g}, dev: {np.abs(est_MC - true_val):.3g} "
          f"(true value: {true_val:.5g})")
    print(f"Estimated value with importance sampling: {est_imp:.4g}, dev: {np.abs(est_imp - true_val_interval):.4g} "
          f"(true value: {true_val_interval})")

run_experiment()