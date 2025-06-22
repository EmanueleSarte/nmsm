
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

double random01() {
    static std::random_device rd;                           // Seed source
    static std::mt19937 gen(rd());                          // Mersenne Twister RNG
    static std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform in [0, 1)
    return dis(gen);
}

struct AppData {
    int n;
    int nsteps;
    double J;

    double T;
    double kB;

    bool save_spins;

    char *memory;
    char **spins;

    double *energies;
    double *magnets;

    ofstream outfile;
};

double get_energy(AppData &dat) {
    double energy = 0;
    for (int i = 0; i < dat.n; i++) {
        for (int j = 0; j < dat.n; j++) {
            energy += dat.spins[i][j] * (dat.spins[i][(j + 1) % dat.n] + dat.spins[(i + 1) % dat.n][j]);
        }
    }
    return energy * dat.J * -1;
}

void generate_configuration(AppData &dat) {
    double r;
    for (int i = 0; i < dat.n; i++) {
        for (int j = 0; j < dat.n; j++) {
            r = random01();
            if (r < 0.5) {
                dat.spins[i][j] = -1;
            } else {
                dat.spins[i][j] = 1;
            }
        }
    }
}

int get_magnetization(AppData &dat) {
    int ma = 0;
    for (int i = 0; i < dat.n; i++) {
        for (int j = 0; j < dat.n; j++) {
            ma += dat.spins[i][j];
        }
    }
    return ma;
}

void save_stats(AppData &dat) {
    dat.outfile.write(reinterpret_cast<const char *>(dat.energies), dat.nsteps * sizeof(dat.energies[0]));
    dat.outfile.write(reinterpret_cast<const char *>(dat.magnets), dat.nsteps * sizeof(dat.magnets[0]));
}

void save_spins(AppData &dat) {
    if (!dat.save_spins) return;

    for (int row = 0; row < dat.n; ++row) {
        dat.outfile.write(reinterpret_cast<const char *>(dat.spins[row]), dat.n * sizeof(char));
    }
}

void save_parameters(AppData &dat) {
    dat.outfile.write(reinterpret_cast<const char *>(&dat.n), sizeof(dat.n));
    dat.outfile.write(reinterpret_cast<const char *>(&dat.nsteps), sizeof(dat.nsteps));
    dat.outfile.write(reinterpret_cast<const char *>(&dat.kB), sizeof(dat.kB));
    dat.outfile.write(reinterpret_cast<const char *>(&dat.T), sizeof(dat.T));
    dat.outfile.write(reinterpret_cast<const char *>(&dat.J), sizeof(dat.J));
    dat.outfile.write(reinterpret_cast<const char *>(&dat.save_spins), sizeof(dat.save_spins));
}

void run_simulation(AppData &dat) {
    save_parameters(dat);

    double energy = get_energy(dat);
    int magnet = get_magnetization(dat);

    dat.energies[0] = energy;
    dat.magnets[0] = magnet;

    save_spins(dat);

    double J = dat.J;
    double kB = dat.kB;
    double T = dat.T;
    char **spins = dat.spins;
    int n = dat.n;
    double exp_values[] = {exp(8 * J / (kB * T)), exp(4 * J / (kB * T)), exp(0 * J / (kB * T)), exp(-4 * J / (kB * T)), exp(-8 * J / (kB * T))};

    for (int m = 1; m < dat.nsteps; m++) {
        // for (int k = 0; k < nx * ny; k++) {
        //     int i = (int)(random01() * nx);
        //     int j = (int)(random01() * ny);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                char neigh_sum = spins[i][(j + 1) % n] + spins[i][(j - 1 + n) % n] + spins[(i + 1) % n][j] + spins[(i - 1 + n) % n][j];
                char offset = neigh_sum * spins[i][j];
                char deltaE = offset * 2;

                if (deltaE < 0 || (random01() < exp_values[(offset + 4) / 2])) {
                    energy += deltaE;
                    magnet += -2 * spins[i][j];

                    spins[i][j] = spins[i][j] * -1;
                }
            }
        }

        dat.energies[m] = energy;
        dat.magnets[m] = magnet;

        save_spins(dat);
    }
    save_stats(dat);
}

void allocate_memory(AppData &dat) {
    dat.memory = new char[dat.n * dat.n];
    dat.spins = new char *[dat.n];

    for (int i = 0; i < dat.n; i++) {
        dat.spins[i] = &dat.memory[i * dat.n];
    }

    dat.energies = new double[dat.nsteps];
    dat.magnets = new double[dat.nsteps];
}

void deallocate_memory(AppData &dat) {
    delete[] dat.memory;
    delete[] dat.spins;

    delete[] dat.energies;
    delete[] dat.magnets;
}

int main(int argc, const char *argv[]) {
    AppData dat;

    dat.nsteps = 200000;
    dat.J = 1;
    dat.kB = 1;
    dat.save_spins = false;

    int nsim = 10;
    double temps[nsim];

    double tc = 2.0 / log(1 + sqrt(2));

    int nval = 11;
    int nvalues[nval] = {
        10, 20, 30, 40,
        50, 60, 70,
        80, 90, 100, 110};
    double ranges[nval] = {
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.065,
        0.05, 0.045, 0.045, 0.045};
    double offsets[nval] = {
        0.055, 0.042, 0.025, 0.025,
        0.025, 0.025, 0.025,
        0.0125, 0.0125, 0.0125, 0.0125
    };

    for (int m = 0; m < nval; m++) {
        // dat.n = (m + 1) * nval;
        dat.n = nvalues[m];

        if (dat.n != 110) continue;

        double dtemp = ranges[m] / nsim;
        for (int i = 0; i < nsim; i++) {
            temps[i] = tc - ranges[m] / 2 + i * dtemp + offsets[m];
        }

        cout << "SIMULATION WITH N=" + to_string(dat.n) << endl;

        std::string filename = "ex_12_N" + to_string(dat.n) + "_M" + to_string(dat.nsteps) + ".bin";
        dat.outfile.open("stats_" + filename, ios::out | ios::binary);
        dat.outfile.write(reinterpret_cast<const char *>(&nsim), sizeof(nsim));

        for (int i = 0; i < nsim; i++) {
            dat.T = temps[i];

            cout << "Running the " + to_string(i) + "-th simulation, with T=" + to_string(dat.T) << endl;

            allocate_memory(dat);

            generate_configuration(dat);
            run_simulation(dat);

            deallocate_memory(dat);
        }

        dat.outfile.close();
    }

    return 0;
}
