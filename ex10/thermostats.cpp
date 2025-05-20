
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
    NormalGenerator(double temp) : rd_(), gen_(rd_()), dis_(0.0, sqrt(temp)) {}

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
    int nbuf;
    int save_par_every;
    double dt;
    double temp;
    double eps;
    double sigma;
    double size[3];

    int thermostat;

    double temp_init;
    double omega;

    double sigma_cut;
    double rho;

    double **pos;
    double **vel;
    double **tpos;
    double *memory;

    double **forces;

    double save_particles;
    ofstream stats_file;
    ofstream particle_file;
};

double random01() {
    static std::random_device rd;                           // Seed source
    static std::mt19937 gen(rd());                          // Mersenne Twister RNG
    static std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform in [0, 1)
    return dis(gen);
}

// double random_normal(double temp) {
//     static std::random_device rd;  // Seed source
//     static std::default_random_engine gen(rd());
//     static std::normal_distribution<double> dis(0, 1.0 / sqrt(temp));
//     return dis(gen);
// }

// Assume the box bottom left corner is (0, 0, 0)
double get_dist2_pbc(AppData &dat, double *pos1, double *pos2, double *dr) {
    dr[0] = pos2[0] - pos1[0];
    dr[1] = pos2[1] - pos1[1];
    dr[2] = pos2[2] - pos1[2];

    dr[0] = dr[0] - round(dr[0] / dat.size[0]) * dat.size[0];
    dr[1] = dr[1] - round(dr[1] / dat.size[1]) * dat.size[1];
    dr[2] = dr[2] - round(dr[2] / dat.size[2]) * dat.size[2];

    return dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
}

void initialize_positions(AppData &dat) {
    for (int i = 0; i < dat.N; i++) {
        dat.pos[i][0] = random01() * dat.size[0];
        dat.pos[i][1] = random01() * dat.size[1];
        dat.pos[i][2] = random01() * dat.size[2];
    }
}

void initialize_velocities(AppData &dat, NormalGenerator &gen) {
    for (int i = 0; i < dat.N; i++) {
        dat.vel[i][0] = gen();
        dat.vel[i][1] = gen();
        dat.vel[i][2] = gen();
    }
}

void initialize_as_lattice(AppData &dat) {
    int p = 0;
    for (int k = 0; (k < 10) && p < dat.N; k++) {

        for (int i = 0; (i < 5) && p < dat.N; i++) {

            for (int j = 0; (j < 4) && p < dat.N; j++) {

                dat.pos[p][0] = (dat.size[0] / 5) * (j + 1);
                dat.pos[p][1] = (dat.size[1] / 5) * (i + 0.5);
                dat.pos[p][2] = (dat.size[2] / 10) * k;

                p += 1;
            }
        }
    }
}

double get_LJ_pot(AppData &dat, double dist2) {
    if (dist2 > dat.sigma_cut * dat.sigma_cut) {
        return 0;
    }

    double ratio = dat.sigma * dat.sigma / dist2;
    return 4 * dat.eps * (pow(ratio, 6) - pow(ratio, 3));
}

void get_energy(AppData &dat, double *ek, double *u) {
    double _ek = 0;
    double _u = 0;
    double tmp[3];
    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < i; j++) {
            double dist2 = get_dist2_pbc(dat, dat.pos[i], dat.pos[j], tmp);
            double pot = get_LJ_pot(dat, dist2);
            _u += pot;
        }
        _ek += 0.5 * (dat.vel[i][0] * dat.vel[i][0] + dat.vel[i][1] * dat.vel[i][1] + dat.vel[i][2] * dat.vel[i][2]);
    }
    *ek = _ek;
    *u = _u;
}

double get_pair_corr(AppData &dat, double r, double dr) {
    double const sigma = 0.02;
    double const fact = 1 / (sigma * sqrt(2 * M_PI));
    double const V = dat.size[0] * dat.size[1] * dat.size[2];

    if (r == 0) {
        return 0;
    }

    double tmp[3];

    double func = 0;
    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < dat.N; j++) {
            if (i == j) continue;

            double dist2 = get_dist2_pbc(dat, dat.pos[i], dat.pos[j], tmp);
            // if ((dist2 > r * r) && (dist2 < (r + dr) * (r + dr))) {
            //     func += 1;
            // }
            double dist = sqrt(dist2);
            double eexp = exp(-(r - dist) * (r - dist) / (2 * sigma * sigma));
            func += eexp * fact;
        }
    }
    double out = func * V / (4 * M_PI * r * r * dat.N * dat.N);
    return out;
}

double get_temperature(AppData &dat) {
    double vsquared = 0;
    for (int i = 0; i < dat.N; i++) {
        vsquared += dat.vel[i][0] * dat.vel[i][0] + dat.vel[i][1] * dat.vel[i][1] + dat.vel[i][2] * dat.vel[i][2];
    }
    return vsquared / (3.0 * dat.N);
}

void rescale_velocities(AppData &dat) {
    double tempnow = get_temperature(dat);
    double lambda = sqrt(dat.temp / tempnow);
    for (int i = 0; i < dat.N; i++) {
        for (int k = 0; k < 3; k++) {
            dat.vel[i][k] = dat.vel[i][k] * lambda;
        }
    }
}

void apply_andersen_thermostat(AppData &dat, NormalGenerator &gen) {
    double prob = dat.omega * dat.dt;
    for (int i = 0; i < dat.N; i++) {
        if (random01() > prob) continue;

        for (int k = 0; k < 3; k++) {
            dat.vel[i][k] = gen();
        }
    }
}

// it's not actually the force, but it's the force divided by r
double get_force_over_r(AppData &dat, double dist2) {
    if (dist2 > dat.sigma_cut * dat.sigma_cut) {
        return 0;
    }

    double ratio = dat.sigma * dat.sigma / dist2;
    return -24 * dat.eps * (2 * pow(ratio, 6) - pow(ratio, 3)) / dist2;
}

void velocity_verlet_sweep(AppData &dat) {
    double dist2, force;
    double dr[3];

    memset(dat.forces[0], 0, sizeof(double) * dat.N * 3);

    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < i; j++) {
            dist2 = get_dist2_pbc(dat, dat.pos[i], dat.pos[j], dr);
            double force = get_force_over_r(dat, dist2);
            for (int k = 0; k < 3; k++) {
                dat.forces[i][k] += force * dr[k];
                dat.forces[j][k] -= force * dr[k];
            }
        }
    }

    for (int i = 0; i < dat.N; i++) {
        for (int k = 0; k < 3; k++) {
            // Save in a different location the new positions, because we'll update them only at the end
            dat.tpos[i][k] = dat.pos[i][k] + dat.vel[i][k] * dat.dt + 0.5 * dat.forces[i][k] * dat.dt * dat.dt;
            dat.tpos[i][k] = dat.tpos[i][k] - floor(dat.tpos[i][k] / dat.size[k]) * dat.size[k];  // PBC
        }
    }

    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < i; j++) {
            dist2 = get_dist2_pbc(dat, dat.tpos[i], dat.tpos[j], dr);
            double force = get_force_over_r(dat, dist2);
            for (int k = 0; k < 3; k++) {
                dat.forces[i][k] += force * dr[k];
                dat.forces[j][k] -= force * dr[k];
            }
        }
    }
    for (int i = 0; i < dat.N; i++) {
        for (int k = 0; k < 3; k++) {
            dat.vel[i][k] = dat.vel[i][k] + 0.5 * dat.dt * dat.forces[i][k];  // f[k] include the old force and the new
        }
    }

    // We have a contiguous chunk of memory with 9*N doubles (3 for pos, 3 for vel, 3 for temp pos)
    // We know memory[0] is the pos[0] and memory[N*6] is the temp pos, so we copy the latter to the former
    memcpy(&dat.memory[0], &dat.memory[dat.N * 6], dat.N * 3 * sizeof(double));
}

void construct_pair_corr(AppData &dat, int iters, double dx, double *values) {
    double end = dat.size[0] / 2;

    for (int i = 0; i < iters; i++) {
        double r = dx * i;
        double g = get_pair_corr(dat, r, dx);
        values[i] = g;
    }
}

void save_params_to_file(AppData &dat) {
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.N), sizeof(dat.N));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.niters), sizeof(dat.niters));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.nbuf), sizeof(dat.nbuf));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.save_par_every), sizeof(dat.save_par_every));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.dt), sizeof(dat.dt));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.eps), sizeof(dat.eps));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sigma), sizeof(dat.sigma));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.rho), sizeof(dat.rho));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.size[0]), sizeof(dat.size[0]));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.size[1]), sizeof(dat.size[0]));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.size[2]), sizeof(dat.size[0]));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sigma_cut), sizeof(dat.sigma_cut));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.omega), sizeof(dat.omega));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.thermostat), sizeof(dat.thermostat));
}

void save_data_to_file(AppData &dat, double *en_k, double *en_u, double *temps) {
    dat.stats_file.write(reinterpret_cast<char *>(en_k), dat.nbuf * sizeof(en_k[0]));
    dat.stats_file.write(reinterpret_cast<char *>(en_u), dat.nbuf * sizeof(en_u[0]));
    dat.stats_file.write(reinterpret_cast<char *>(temps), dat.nbuf * sizeof(temps[0]));
}

void save_particle_to_file(AppData &dat) {
    if (!dat.save_particles) return;
    dat.particle_file.write(reinterpret_cast<char *>(dat.memory), dat.N * 6 * sizeof(dat.memory[0]));
}

void run_simulation(AppData &dat) {
    save_params_to_file(dat);
    save_particle_to_file(dat);

    NormalGenerator gen(dat.temp);

    int nbuf = dat.nbuf;
    double *en_k = new double[nbuf];
    double *en_u = new double[nbuf];
    double *temps = new double[nbuf];

    int n_gsave = 50;
    int ng = 200;
    double dx = (dat.size[0] / 2) / ng;
    double *pair_corr = new double[ng * n_gsave];

    get_energy(dat, &en_k[0], &en_u[0]);
    temps[0] = get_temperature(dat);
    cout << "Initial Energy: " << en_k[0] + en_u[0] << endl;

    for (int i = 1; i < dat.niters; i++) {
        velocity_verlet_sweep(dat);

        if (dat.thermostat == 1) {
            rescale_velocities(dat);

        } else if (dat.thermostat == 2) {
            apply_andersen_thermostat(dat, gen);
        }

        get_energy(dat, &en_k[i % nbuf], &en_u[i % nbuf]);
        temps[i % nbuf] = get_temperature(dat);

        if ((i + 1) % nbuf == 0) {
            save_data_to_file(dat, en_k, en_u, temps);
        }

        if (i % dat.save_par_every == 0) {
            save_particle_to_file(dat);
        }

        if (i >= (dat.niters - n_gsave)) {
            int idx = i + n_gsave - dat.niters;
            construct_pair_corr(dat, ng, dx, &pair_corr[idx * ng]);
        }
    }

    dat.stats_file.write(reinterpret_cast<const char *>(&n_gsave), sizeof(ng));
    dat.stats_file.write(reinterpret_cast<const char *>(&ng), sizeof(ng));
    dat.stats_file.write(reinterpret_cast<const char *>(&dx), sizeof(dx));
    dat.stats_file.write(reinterpret_cast<char *>(pair_corr), n_gsave * ng * sizeof(pair_corr[0]));

    delete[] en_k;
    delete[] en_u;
    delete[] temps;
    delete[] pair_corr;
}

void allocate_memory(AppData &dat) {
    double *memory = new double[dat.N * (3 + 3 + 3 + 3)];

    dat.memory = memory;
    dat.pos = new double *[dat.N];
    dat.vel = new double *[dat.N];
    dat.tpos = new double *[dat.N];
    dat.forces = new double *[dat.N];
    for (int i = 0; i < dat.N; i++) {
        dat.pos[i] = &memory[i * 3];
        dat.vel[i] = &memory[3 * dat.N + i * 3];
        dat.tpos[i] = &memory[6 * dat.N + i * 3];
        dat.forces[i] = &memory[9 * dat.N + i * 3];
    }
}

void deallocate_memory(AppData &dat) {
    delete[] dat.memory;
    delete[] dat.pos;
    delete[] dat.vel;
    delete[] dat.tpos;
    delete[] dat.forces;
}

void initialize_data(AppData &dat) {
    dat.sigma = 1;
    double side = 10 * dat.sigma;
    double vol = side * side * side;
    dat.rho = 0.2 * pow(dat.sigma, -3);

    dat.N = (int)round(vol * dat.rho);

    dat.dt = 0.01;
    dat.niters = 100000;
    dat.nbuf = 1000;
    dat.save_par_every = 1000;
    dat.temp = 1;
    dat.eps = 1;

    dat.size[0] = side;
    dat.size[1] = side;
    dat.size[2] = side;

    dat.sigma_cut = 4;
    dat.save_particles = true;
    dat.temp_init = 1;

    dat.omega = 0.05 / dat.dt;
}

void run_nve_simulation(AppData &dat, int nsc, double *sigma_cuts) {
    NormalGenerator gen_init(dat.temp_init);

    for (int i = 0; i < nsc; i++) {
        allocate_memory(dat);

        dat.sigma_cut = sigma_cuts[i];

        cout << "Sim: " << i << ", TH_" << dat.thermostat << ", N=" << dat.N << ", M=" << dat.niters << ", sigma_cut=" << dat.sigma_cut << endl;

        initialize_as_lattice(dat);
        initialize_velocities(dat, gen_init);
        run_simulation(dat);

        deallocate_memory(dat);
    }
}

void run_nvt_simulation(AppData &dat, int ns, int *nparticles) {
    NormalGenerator gen_init(dat.temp_init);

    for (int i = 0; i < ns; i++) {
        dat.N = nparticles[i];
        allocate_memory(dat);

        cout << "Sim: " << i << ", TH_" << dat.thermostat << ", N=" << dat.N << ", M=" << dat.niters << ", sigma_cut=" << dat.sigma_cut << endl;

        initialize_as_lattice(dat);
        initialize_velocities(dat, gen_init);
        run_simulation(dat);

        deallocate_memory(dat);
    }
}

int main(int argc, char const *argv[]) {
    AppData dat;
    initialize_data(dat);
    dat.thermostat = 0;  // 0 No th, 1 = V rescaling, 2 = Anderson

    int nsc = 16;
    double sigma_cuts[nsc] = {pow(2, 1.0 / 6), 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4};

    // int nsc = 1;
    // double sigma_cuts[nsc] = {4};

    int npart = 8;
    int particles[npart] = {25, 50, 75, 100, 125, 150, 175, 200};

    int nsim = (dat.thermostat == 0) ? nsc : npart;
    string filename = "ex10_TH" + to_string(dat.thermostat) + "_NSIM" + to_string(nsim) + "_M" + to_string(dat.niters) + ".bin";

    dat.stats_file.open("stats_" + filename, ios::out | ios::binary);
    if (dat.save_particles) dat.particle_file.open("particles_" + filename, ios::out | ios::binary);

    dat.stats_file.write(reinterpret_cast<const char *>(&nsim), sizeof(nsim));

    if (dat.thermostat == 0) {
        dat.temp = 1;
        dat.N = 200;
        run_nve_simulation(dat, nsc, sigma_cuts);
    } else {
        dat.temp = 2;
        run_nvt_simulation(dat, npart, particles);
    }

    dat.stats_file.close();
    if (dat.save_particles) dat.particle_file.close();

    return 0;
}
