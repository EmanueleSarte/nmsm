import matplotlib.pyplot as plt
import numpy as np

np.seterr(all='raise')


def get_acceleration(q, omega):
    return -q * omega ** 2


def velocity_verlet(curr_state, omega, dt):
    next_state = np.zeros(2)
    q = curr_state[0]
    p = curr_state[1]
    acc_q = get_acceleration(q, omega)
    next_state[0] = q + p * dt + 0.5 * acc_q * dt ** 2
    next_state[1] = p + 0.5 * dt * (acc_q + get_acceleration(next_state[0], omega))
    return next_state


def beeman_alg(prev_state, curr_state, omega, dt):
    next_state = np.zeros(2)
    q = curr_state[0]
    p = curr_state[1]

    aprev = get_acceleration(prev_state[0], omega)
    anow = get_acceleration(curr_state[0], omega)

    next_state[0] = q + p * dt + (1 / 6) * (4 * anow - aprev) * (dt ** 2)
    next_state[1] = p + (1 / 6) * (2 * get_acceleration(next_state[0], omega) + 5 * anow - aprev) * dt

    return next_state


def exact_solution(times, A, omega, phi):
    try:
        states = np.zeros((len(times), 2))
    except Exception as e:
        states = np.zeros(2)

    states[..., 0] = A * np.sin(omega * times + phi)
    states[..., 1] = A * np.cos(omega * times + phi) * omega
    return states


def run_velocity_verlet(niters, dt, omega, q0, p0):
    state_history = np.zeros((niters, 2))
    state_history[0] = [q0, p0]
    for i in range(niters - 1):
        nstate = velocity_verlet(state_history[i], omega, dt)
        state_history[i + 1] = nstate

    return state_history


def run_beeman_alg(niters, dt, omega, q0, p0):
    state_history = np.zeros((niters, 2))
    state_minus1 = exact_solution(-dt, 1 / omega, omega, 0)
    state_history[0] = [q0, p0]
    # state_history[1] = [q0, p0]

    for i in range(0, niters - 1):
        if i == 0:
            nstate = beeman_alg(state_minus1, state_history[0], omega, dt)
        else:
            nstate = beeman_alg(state_history[i - 1], state_history[i], omega, dt)
        state_history[i + 1] = nstate

    return state_history


def get_energy(state, omega):
    return 0.5 * (omega ** 2) * (state[..., 0] ** 2) + 0.5 * state[..., 1] ** 2


def plot_energy_conservation(e0, omega, energy_vv, energy_bm, dt):
    deltae_vv = (energy_vv - e0) / e0
    deltae_bm = (energy_bm - e0) / e0
    time = np.arange(len(deltae_vv)) * dt

    plt.figure()
    plt.title(r"$(E(t)-E_0)/E_0$, $\omega$=" + f"{omega}, " + r" $\Delta t$=" + f"{dt}, steps={len(energy_vv)}")
    plt.plot(time, deltae_vv, label="Velocity Verlet")
    plt.plot(time, deltae_bm, label="Beeman's Alg.")
    plt.xlabel("time")
    plt.ylabel("Energy difference")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_solutions(times, states_exact, states_vv, states_bm=None):
    plt.figure()
    plt.plot(times, states_exact[:, 0], label="Exact Solution")
    plt.plot(times, states_vv[:, 0], label="Velocity Verlet")
    if states_bm is not None:
        plt.plot(times, states_bm[:, 0], label="Beeman's Alg")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_discrepancy(times, omega, states_exact, states_vv, states_bm=None, var=1):
    plt.figure()
    dt = times[1] - times[0]
    if var == 1:
        plt.plot(times, (states_vv[:, 0] - states_exact[:, 0]), label="Velocity Verlet")
        if states_bm is not None:
            plt.plot(times, states_bm[:, 0] - states_exact[:, 0], label="Beeman's Alg")
        plt.ylabel("$q(t) - q_{exact}(t)$")
        plt.title(r"$q(t) - q_{exact}(t)$, $\omega$=" + f"{omega}, " + r" $\Delta t$=" + f"{dt}, steps={len(times)}")
    else:
        plt.plot(times, (states_vv[:, 1] - states_exact[:, 1]), label="Velocity Verlet")
        if states_bm is not None:
            plt.plot(times, states_bm[:, 1] - states_exact[:, 1], label="Beeman's Alg")
        plt.ylabel("$p(t) - p_{exact}(t)$")
        plt.title(r"$p(t) - p_{exact}(t)$, $\omega$=" + f"{omega}, " + r" $\Delta t$=" + f"{dt}, steps={len(times)}")
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_stability(dt, omega, states_vv):
    plt.figure(figsize=(5.5, 5.5))

    times = np.linspace(0, 2 * np.pi / omega, 10000)
    plt.plot(1 / omega * np.sin(omega * times), np.cos(omega * times), label="Exact Solution",
             linestyle=":", zorder=10, color="red")

    mult = 10
    index = round(mult * 2 * np.pi / omega / dt) + 1
    plt.plot(states_vv[-index:, 0], states_vv[-index:, 1], label="Velocity Verlet")
    plt.title(r"Velocity Verlet Stability, $\omega$=" + f"{omega}, " + r" $\Delta t$=" + f"{dt}, " +
              r"$N_{10rev}$" + f"={index}")
    plt.xlabel("$q(t)$")
    plt.ylabel("$p(t)$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_simulation():
    niters = 100000
    dt = 0.001
    q0, p0 = 0, 1
    omega = 1
    e0 = 0.5 * q0 * (omega ** 2) + 0.5 * (p0 ** 2)

    vvres = run_velocity_verlet(niters, dt, omega, q0, p0)
    energy_vv = get_energy(vvres, omega)

    bmres = run_beeman_alg(niters, dt, omega, q0, p0)
    energy_bm = get_energy(bmres, omega)

    times = np.arange(niters) * dt
    exactres = exact_solution(times, 1 / omega, omega, 0)

    # plot_solutions(times, exactres, vvres, bmres)
    plot_energy_conservation(e0, omega, energy_vv, energy_bm, dt)
    plot_discrepancy(times, omega, exactres, vvres, bmres, var=1)
    plot_discrepancy(times, omega, exactres, vvres, bmres, var=2)
    # plot_stability(vvres)


def run_stability_check_vv():
    niters = 100
    dt = 0.01
    q0, p0 = 0, 1

    for omega in [1, 0.1, 1, 10, 100]:
        e0 = 0.5 * q0 * (omega ** 2) + 0.5 * (p0 ** 2)
        vvres = run_velocity_verlet(niters, dt, omega, q0, p0)
        times = np.arange(niters) * dt
        exactres = exact_solution(times, 1 / omega, omega, 0)
        # plot_discrepancy(times, omega, exactres, vvres, var=1)
        plot_stability(dt, omega, vvres)


# run_simulation()
run_stability_check_vv()
