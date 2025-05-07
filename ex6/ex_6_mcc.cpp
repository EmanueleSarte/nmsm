
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

struct AppParams {
    double ***polymers;
    int ktot;
    int n;
    int niters;
    double *betas;
    double c;
    double maxc;
    double maxtheta;
    double h;
    double eps;
    int polymer_skip_save;

    int shift_per_block;
    int block_per_sample;

    bool save_polymers;

    double **aux_points;

    ofstream poly_file;
    ofstream stats_file;
};

double random01() {
    static std::random_device rd;                           // Seed source
    static std::mt19937 gen(rd());                          // Mersenne Twister RNG
    static std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform in [0, 1)
    return dis(gen);
}

void initialize_polymer(double **polymer, int n) {
    for (int i = 1; i < n; i++) {
        polymer[i][0] = 0;
        polymer[i][1] = i;
    }
}

int get_energy(double **polymer, int n, double h, int start, int end) {
    int energy = 0;
    for (int i = start; i < end; i++) {
        if (polymer[i][1] < h) {
            energy -= 1;
        }
    }
    return energy;
}

int get_energy(double **polymer, int n, double h) {
    return get_energy(polymer, n, h, 1, n);
}

// Assuming i > 0
bool is_shift_legal(double **polymer, int n, int i, double c, double maxc, double deltax, double deltay) {
    for (int j = 0; j < n; j++) {
        if (j == i) {
            if ((polymer[i][1] + deltay - c / 2) < 0) {  // Check if going into the wall
                return false;
            }
        } else {
            double dist = pow(polymer[j][0] - polymer[i][0] - deltax, 2) + pow(polymer[j][1] - polymer[i][1] - deltay, 2);
            if (dist < c) {  // check for overlapping
                return false;
            }
            if (j == (i - 1) || j == (i + 1)) {  // check for bond lenght
                if (dist > maxc) {
                    return false;
                }
            }
        }
    }
    return true;
}

// The new position for monomere i will be in newpos[i][0/1]
bool is_pivot_legal(double **polymer, int n, int i, double c, double maxc, double **newpos) {
    for (int j = 0; j <= i; j++) {         // For every monomere before and at the pivot, check if the distance with
        for (int k = i + 1; k < n; k++) {  // the monomere at or after the pivot
            double dist = pow(polymer[j][0] - newpos[k][0], 2) + pow(polymer[j][1] - newpos[k][1], 2);
            if (dist < c) {  // is overlapping
                return false;
            }
        }
    }
    for (int k = i + 1; k < n; k++) {  // check if it is going into the wall
        if ((newpos[k][1] - c / 2) < 0) {
            return false;
        }
    }

    return true;
}

void rotate_points(double **polymer, int n, int i, double angle, double **final_points) {
    double cos_theta = cos(angle);
    double sin_theta = sin(angle);
    double cx = polymer[i][0];
    double cy = polymer[i][1];
    for (int j = i + 1; j < n; j++) {
        final_points[j][0] = ((polymer[j][0] - cx) * cos_theta - (polymer[j][1] - cy) * sin_theta) + cx;
        final_points[j][1] = ((polymer[j][0] - cx) * sin_theta + (polymer[j][1] - cy) * cos_theta) + cy;
    }
}

void propose_shift(double c, double *deltax, double *deltay) {
    *deltax = (random01() - 0.5) * 2 * c * 0.75;
    *deltay = (random01() - 0.5) * 2 * c * 0.75;
}

void propose_pivot(double maxtheta, double *theta) {
    *theta = (random01() - 0.5) * 2 * maxtheta;
}

void save_polymer(AppParams &par, int k) {
    if (!par.save_polymers) return;
    // par.outfile.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
    double **polymer = par.polymers[k];
    for (int i = 0; i < par.n; i++) {
        par.poly_file.write(reinterpret_cast<const char *>(&polymer[i][0]), sizeof(polymer[i][0]));
        par.poly_file.write(reinterpret_cast<const char *>(&polymer[i][1]), sizeof(polymer[i][1]));
    }
}

void save_stats(AppParams &par, int *energies) {
    for (int k = 0; k < par.ktot; k++) {
        double en = energies[k] * 1.0;
        double **poly = par.polymers[k];
        double dist2 = pow(poly[par.n - 1][0], 2) + pow(poly[par.n - 1][1], 2);

        par.stats_file.write(reinterpret_cast<const char *>(&en), sizeof(en));
        par.stats_file.write(reinterpret_cast<const char *>(&dist2), sizeof(dist2));
        par.stats_file.write(reinterpret_cast<const char *>(&poly[par.n - 1][1]), sizeof(poly[par.n - 1][1]));
    }
}

void save_swap_stats(AppParams &par, int *swap_stats) {
    for (int i = 0; i < par.ktot; i++) {
        par.stats_file.write(reinterpret_cast<const char *>(&swap_stats[i]), sizeof(swap_stats[0]));
        par.stats_file.write(reinterpret_cast<const char *>(&swap_stats[i + par.ktot]), sizeof(swap_stats[0]));
        par.stats_file.write(reinterpret_cast<const char *>(&swap_stats[i + par.ktot * 2]), sizeof(swap_stats[0]));
        par.stats_file.write(reinterpret_cast<const char *>(&swap_stats[i + par.ktot * 3]), sizeof(swap_stats[0]));
    }
}

void save_parameters(AppParams &par) {
    par.stats_file.write(reinterpret_cast<const char *>(&par.niters), sizeof(par.niters));
    par.stats_file.write(reinterpret_cast<const char *>(&par.n), sizeof(par.n));
    par.stats_file.write(reinterpret_cast<const char *>(&par.c), sizeof(par.c));
    par.stats_file.write(reinterpret_cast<const char *>(&par.maxc), sizeof(par.maxc));
    par.stats_file.write(reinterpret_cast<const char *>(&par.maxtheta), sizeof(par.maxtheta));
    par.stats_file.write(reinterpret_cast<const char *>(&par.h), sizeof(par.h));
    par.stats_file.write(reinterpret_cast<const char *>(&par.ktot), sizeof(par.ktot));
    par.stats_file.write(reinterpret_cast<const char *>(&par.save_polymers), sizeof(par.save_polymers));
    par.stats_file.write(reinterpret_cast<const char *>(&par.polymer_skip_save), sizeof(par.polymer_skip_save));

    for (int i = 0; i < par.ktot; i++) {
        par.stats_file.write(reinterpret_cast<const char *>(&par.betas[i]), sizeof(par.betas[i]));
    }
}

bool metropolis_shift(AppParams &par, int k, int *deltaE) {
    double deltax, deltay;
    propose_shift(par.c, &deltax, &deltay);

    double **polymer = par.polymers[k];
    int mono = (int)(random01() * (par.n - 1) + 1);
    if (is_shift_legal(polymer, par.n, mono, par.c, par.maxc, deltax, deltay)) {
        int local_deltaE = ((polymer[mono][1] + deltay < par.h) ? -1 : 0) - ((polymer[mono][1] < par.h) ? -1 : 0);

        if (local_deltaE <= 0 || random01() < exp(-par.betas[k] * local_deltaE * par.eps)) {
            polymer[mono][0] += deltax;
            polymer[mono][1] += deltay;

            *deltaE = local_deltaE;
            return true;
        }
    }
    return false;
}

bool metropolis_pivot(AppParams &par, int k, int *deltaE) {
    double theta;
    propose_pivot(par.maxtheta, &theta);

    int n = par.n;
    double **polymer = par.polymers[k];
    int mono = (int)(random01() * (n - 1));  // Possible pivots: [0 to n-2] (not the last one)
    rotate_points(polymer, n, mono, theta, par.aux_points);

    if (is_pivot_legal(polymer, n, mono, par.c, par.maxc, par.aux_points)) {
        int local_deltaE = get_energy(par.aux_points, n, par.h, mono + 1, n) - get_energy(polymer, n, par.h, mono + 1, n);

        if (local_deltaE < 0 || random01() < exp(-par.betas[k] * local_deltaE * par.eps)) {
            for (int j = mono + 1; j < n; j++) {
                polymer[j][0] = par.aux_points[j][0];
                polymer[j][1] = par.aux_points[j][1];
            }

            *deltaE = local_deltaE;
            return true;
        }
    }
    return false;
}

bool metropolis_swap_chains(AppParams &par, int k, int *energies, int *swap_on_pos, int *swap_on_zero, int *swap_on_perc) {
    int deltaE = energies[k + 1] - energies[k];

    // if (deltaE == 0) return false;

    bool swap = false;
    if (deltaE > 0) {
        swap = true;
        swap_on_pos[k] += 1;
    } else if (deltaE == 0) {
        swap = true;
        swap_on_zero[k] += 1;
    } else if (random01() < exp((par.betas[k + 1] - par.betas[k]) * deltaE * par.eps)) {
        swap = true;
        swap_on_perc[k] += 1;
    }

    if (swap) {
        double **tmp = par.polymers[k];
        par.polymers[k] = par.polymers[k + 1];
        par.polymers[k + 1] = tmp;

        int en = energies[k];
        energies[k] = energies[k + 1];
        energies[k + 1] = en;
        return true;
    }

    return false;
}

void chain_block_move(AppParams &par, int k, int *energy) {
    int deltaE;
    for (int j = 0; j < par.shift_per_block; j++) {
        if (metropolis_shift(par, k, &deltaE)) {
            *energy += deltaE;
        }
    }

    if (metropolis_pivot(par, k, &deltaE)) {
        *energy += deltaE;
    }
}

void run_simulation(AppParams &par) {
    save_parameters(par);

    int kt = par.ktot;
    int energies[kt];
    for (int k = 0; k < kt; k++) {
        energies[k] = get_energy(par.polymers[k], par.n, par.h);
        save_polymer(par, k);
    }
    save_stats(par, energies);

    int swap_stats[kt * 4] = {0};
    int deltaE;
    for (int i = 1; i < par.niters; i++) {  // we start from 1 since the zero time is the initial config and saved above
        for (int j = 0; j < par.block_per_sample; j++) {
            for (int k = 0; k < kt; k++) {
                chain_block_move(par, k, &energies[k]);
            }
        }

        if (kt > 1) {
            int k = (int)(random01() * (kt - 1));
            metropolis_swap_chains(par, k, energies, &swap_stats[kt * 1], &swap_stats[kt * 2], &swap_stats[kt * 3]);
            swap_stats[k] += 1;
        }

        if ((i % par.polymer_skip_save) == 0) {
            for (int k = 0; k < kt; k++) {
                save_polymer(par, k);
            }
        }

        save_stats(par, energies);
    }
    // cout << "Ho swappato " << swap_done << " su " << par.niters << endl;
    for (int k = 0; k < kt - 1; k++) {
        cout << k << "=" << swap_stats[kt * 1 + k] << "+" << swap_stats[kt * 2 + k] << "+" << swap_stats[kt * 3 + k] << "/" << swap_stats[k] << "  ";
    }
    cout << endl;

    save_swap_stats(par, swap_stats);
    if (par.save_polymers) {
        par.poly_file.close();
    }
    par.stats_file.close();
}

int main(int argc, char const *argv[]) {
    int N = 25;
    int ntot = N + 1;

    AppParams par;
    par.n = ntot;
    par.h = 1;
    par.c = 1;
    par.maxc = 1.3;
    par.ktot = 5;
    par.maxtheta = 3.1415 / 8;
    par.niters = 10000;
    par.shift_per_block = 10;
    par.block_per_sample = 100;
    par.eps = 1;
    par.polymer_skip_save = 1;
    par.save_polymers = true;

    double ***polymers = new double **[par.ktot];
    double *betas = new double[par.ktot];
    for (int k = 0; k < par.ktot; ++k) {
        polymers[k] = new double *[par.n];
        for (int j = 0; j < par.n; ++j) {
            polymers[k][j] = new double[2];
        }
        initialize_polymer(polymers[k], par.n);
    }

    // for (int k = par.ktot - 1; k >= 0; k--) {
    //     double T = 0.25 + k * 0.125;
    //     betas[par.ktot - k - 1] = 1 / T;
    // }

    cout << "Betas: " << "  ";
    for (int k = 0; k < par.ktot; k++) {
        // betas[k] = 2.0 / 3 + 0.12 * k;

        // betas[k] = 0.6 + k * 0.2;
        // betas[k] = 0.5 + k * 0.5;
        // betas[k] = 0.1 + k * 0.1;
        betas[k] = 0.5 + k * 0.75;
        cout << betas[k] << "  ";
    }
    cout << endl;

    par.polymers = polymers;
    par.betas = betas;

    double **aux_points = new double *[par.n];
    for (int i = 0; i < par.n; i++) {
        aux_points[i] = new double[2];
    }
    par.aux_points = aux_points;

    string filename = "_n=" + to_string(par.n) + "_h=" + to_string(par.h) + "_ktot=" + to_string(par.ktot) + "_iters=" + to_string(par.niters) +
                      "_minb=" + to_string(betas[0]) + "_maxb=" + to_string(betas[par.ktot - 1]) + ".bin";

    par.stats_file.open("stats" + filename, ios::out | ios::binary);

    if (par.save_polymers) {
        par.poly_file.open("poly" + filename, ios::out | ios::binary);
    }

    run_simulation(par);

    for (int k = 0; k < par.ktot; ++k) {
        for (int j = 0; j < par.n; ++j) {
            delete[] polymers[k][j];
        }
        delete[] polymers[k];
    }
    delete[] polymers;
    delete[] betas;

    for (int i = 0; i < par.n; i++) {
        delete[] aux_points[i];
    }

    return 0;
}
