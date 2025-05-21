
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

class NormalGenerator {
   public:
    NormalGenerator(double dt) : rd_(), gen_(rd_()), dis_(0.0, sqrt(dt)) {}

    double operator()() {
        return dis_(gen_);
    }

   private:
    std::random_device rd_;
    std::default_random_engine gen_;
    std::normal_distribution<double> dis_;
};

struct AppData {
    int N;
    int niters;
    double dt;
    int ncellside;

    double side;
    double temp;
    double eps;
    int ncell;

    double cellside;
    double r1;
    double r2;
    double D1;
    double D2;
    double mu1;
    double mu2;
    double density;

    double **pos0;
    double **pos;
    double **forces;
    double *memory;

    int *tpar;
    bool *tag;
    int *head;
    int *plist;

    ofstream stats_file;
};

// double random_normal() {
//     static std::random_device rd;  // Seed source
//     static std::default_random_engine gen(rd());
//     static std::normal_distribution<double> dis(0, 1);
//     return dis(gen);
// }

double get_dist2(double *pos1, double *pos2, double *dr) {
    dr[0] = pos2[0] - pos1[0];
    dr[1] = pos2[1] - pos1[1];
    return dr[0] * dr[0] + dr[1] * dr[1];
}

double get_force_over_x(double dist2, double sigma, double eps, double cutoff2) {
    // if (dist2 > cutoff2) return 0;

    return eps * exp(-dist2 / (2 * sigma * sigma));
}

void compute_cell_list(AppData &dat) {
    memset(dat.tag, false, dat.ncell * sizeof(bool));
    for (int i = 0; i < dat.N; i++) {
        int cell = floor(dat.pos[i][0] / dat.cellside) + dat.ncellside * floor(dat.pos[i][1] / dat.cellside);

        if (dat.tag[cell] == false) {
            dat.head[cell] = i;
            dat.plist[i] = -1;
            dat.tag[cell] = true;
        } else {
            dat.plist[i] = dat.head[cell];
            dat.head[cell] = i;
        }
    }
}

void compute_forces(AppData &dat) {
    // Aggiorniamo le forze di tutti in un colpo solo, perchè così partiamo da una cella, aggiorniamo le forze di tutte le particelle
    // di quella cella dato che così evitiamo di visitare robe in più

    static int offsets[5] = {0, 1, dat.ncellside - 1, dat.ncellside, dat.ncellside + 1};
    double dr[2];
    int ncell = pow(dat.ncellside, 2);

    memset(dat.forces[0], 0, sizeof(double) * dat.N * 2);

    // For every cell c1
    for (int c1 = 0; c1 < ncell; c1++) {
        if (dat.tag[c1] == false) continue;

        // For every lower right neighbors cell c2 of cell c1
        for (int idx2 = 0; idx2 < 5; idx2++) {
            int c2 = (c1 + offsets[idx2]) % ncell;
            if (dat.tag[c2] == false) continue;

            // For every particle of cell c1
            int i = dat.head[c1];
            do {
                // For every particle of cell c2
                int j = dat.head[c2];
                do {
                    if (i != j) {
                        double dist2 = get_dist2(dat.pos[i], dat.pos[j], dr);
                        double rtot = (dat.tpar[i] == 1 ? dat.r1 : dat.r2) + (dat.tpar[j] == 1 ? dat.r1 : dat.r2);

                        if (dist2 <= 16 * rtot * rtot) {
                            double force = get_force_over_x(dist2, rtot, dat.eps, 16 * rtot * rtot);

                            for (int k = 0; k < 2; k++) {
                                dat.forces[i][k] += -force * dr[k];
                                dat.forces[j][k] += force * dr[k];
                            }
                        }
                    }

                    j = dat.plist[j];
                    // cout << "fineciclo2" << endl;
                } while (j != -1);

                i = dat.plist[i];
                // cout << "fineciclo1" << endl;
            } while (i != -1);
        }
    }
    // cout << "endlstop" << endl;
}

void update_position(AppData &dat, NormalGenerator &gen) {
    static const double foo[3] = {-1, sqrt(2 * dat.D1), sqrt(2 * dat.D2)};

    for (int i = 0; i < dat.N; i++) {
        for (int k = 0; k < 2; k++) {
            dat.pos[i][k] += dat.mu1 * dat.forces[i][k] * dat.dt + foo[dat.tpar[i]] * gen();

            // Reflective boundary
            if (dat.pos[i][k] > dat.side) {
                double x = dat.pos[i][k];
                if (x > (dat.side * 2)) {
                    cout << "Bigger than 2L: " << x << endl;
                }
                dat.pos[i][k] = x - floor(x / dat.side) * 2 * (x - dat.side);

            } else if (dat.pos[i][k] < 0) {
                double x = dat.pos[i][k];
                if (x < -dat.side) {
                    cout << "Lower than -L: " << x << endl;
                }
                dat.pos[i][k] = x + floor(x / dat.side) * 2 * x;
            }
        }
    }
}

void save_params_to_file(AppData &dat) {
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.N), sizeof(dat.N));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.niters), sizeof(dat.niters));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.dt), sizeof(dat.dt));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.cellside), sizeof(dat.cellside));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.ncellside), sizeof(dat.ncellside));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.side), sizeof(dat.side));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.temp), sizeof(dat.temp));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.eps), sizeof(dat.eps));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.ncell), sizeof(dat.ncell));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.r1), sizeof(dat.r1));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.r2), sizeof(dat.r2));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.D1), sizeof(dat.D1));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.D2), sizeof(dat.D2));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.mu1), sizeof(dat.mu1));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.mu2), sizeof(dat.mu2));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.density), sizeof(dat.density));
    dat.stats_file.write(reinterpret_cast<char *>(dat.tpar), dat.N * sizeof(dat.tpar[0]));
}

void save_particle_to_file(AppData &dat) {
    dat.stats_file.write(reinterpret_cast<char *>(dat.memory), dat.N * 2 * sizeof(dat.memory[0]));
}

void run_simulation(AppData &dat) {
    NormalGenerator gen(dat.dt);
    save_params_to_file(dat);
    save_particle_to_file(dat);

    for (int i = 1; i < dat.niters; i++) {
        // cout << "qui1" << endl;
        compute_cell_list(dat);
        // cout << "qui2" << endl;
        // cout << i << endl;
        compute_forces(dat);
        // cout << "qui3" << endl;
        update_position(dat, gen);
        // cout << "qui4" << endl;
        save_particle_to_file(dat);
        // cout << "qui5" << endl;
    }
}

void initialize_data(AppData &dat, int N, int niters, double dt, double ncellside) {
    dat.N = N;
    dat.niters = niters;
    dat.dt = dt;
    dat.ncellside = ncellside;

    dat.ncell = dat.ncellside * dat.ncellside;
    dat.side = 100;
    dat.r1 = 1.25;
    dat.r2 = 1;
    dat.temp = 1;
    dat.eps = 10;

    dat.cellside = dat.side / dat.ncellside;
    dat.mu1 = 1 / dat.r1;
    dat.mu2 = 1 / dat.r2;
    dat.D1 = dat.mu1 * dat.temp;
    dat.D2 = dat.mu2 * dat.temp;
    dat.density = dat.N / (dat.side * dat.side);
}

void allocate_memory(AppData &dat) {
    double *memory = new double[dat.N * (2 + 2 + 2)];

    dat.memory = memory;
    dat.pos = new double *[dat.N];
    dat.forces = new double *[dat.N];
    dat.pos0 = new double *[dat.N];
    for (int i = 0; i < dat.N; i++) {
        dat.pos[i] = &memory[i * 2];
        dat.forces[i] = &memory[2 * dat.N + i * 2];
        dat.pos0[i] = &memory[4 * dat.N + i * 2];
    }

    dat.tpar = new int[dat.N];

    dat.tag = new bool[dat.ncell];
    dat.head = new int[dat.ncell];
    dat.plist = new int[dat.N];

    memset(dat.tag, false, dat.ncell * sizeof(bool));
}

void deallocate_memory(AppData &dat) {
    delete[] dat.memory;
    delete[] dat.pos;
    delete[] dat.forces;
    delete[] dat.pos0;

    delete[] dat.tpar;
    delete[] dat.tag;
    delete[] dat.head;
    delete[] dat.plist;
}

void assign_particles_type(AppData &dat) {
    for (int i = 0; i < dat.N / 2; i++) {
        dat.tpar[i] = 1;
    }
    for (int i = dat.N / 2; i < dat.N; i++) {
        dat.tpar[i] = 2;
    }
}

void assign_particle_pos(AppData &dat) {
    int k = 0;
    int nstep = (int)ceil(sqrt(dat.N));
    double spacing = dat.side / nstep;

    for (int i = 0; (i < nstep) && (k < dat.N); i++) {
        for (int j = 0; (j < nstep) && (k < dat.N); j++) {
            dat.pos[k][0] = i * spacing + spacing * 0.5;
            dat.pos[k][1] = j * spacing + spacing * 0.5;
            k += 1;
        }
    }
    memcpy(&dat.pos0[0][0], &dat.pos[0][0], dat.N * 2 * sizeof(double));
}

int main(int argc, char const *argv[]) {
    AppData dat;

    int N = 20;
    int niters = 200000;
    double dt = 0.1;
    int ncellside = 10;
    initialize_data(dat, N, niters, dt, ncellside);

    string filename = "ex11_N" + to_string(N) + "_M" + to_string(dat.niters) + ".bin";
    dat.stats_file.open("stats_" + filename, ios::out | ios::binary);
    int nsim = 1;
    dat.stats_file.write(reinterpret_cast<const char *>(&nsim), sizeof(nsim));

    allocate_memory(dat);
    assign_particles_type(dat);
    assign_particle_pos(dat);

    run_simulation(dat);

    dat.stats_file.close();
    deallocate_memory(dat);
    return 0;
}
