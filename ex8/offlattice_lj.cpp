
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

const bool DEBUG = false;

struct AppData {
    int N;
    int niters;
    int nbuf;
    int save_par_every;
    double temp;
    double eps;
    double sigma;
    double dmax;
    double sizex;
    double sizey;
    double sizez;
    double beta;

    double sigma_cut;
    double utail;
    double ptail;
    double rho;

    double *posx;
    double *posy;
    double *posz;

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

// Assume the box bottom left corner is (0, 0, 0)
double get_dist2_pbc(AppData &dat, double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;

    dx = dx - round(dx / dat.sizex) * dat.sizex;
    dy = dy - round(dy / dat.sizey) * dat.sizey;
    dz = dz - round(dz / dat.sizez) * dat.sizez;

    return dx * dx + dy * dy + dz * dz;
}

double get_LJ_pot(AppData &dat, double dist2) {
    if (dist2 > dat.sigma_cut * dat.sigma_cut) {
        return 0;
    }

    double ratio = dat.sigma * dat.sigma / dist2;
    return 4 * dat.eps * (pow(ratio, 6) - pow(ratio, 3));
}

double get_energy(AppData &dat) {
    double energy = 0;
    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < i; j++) {
            double dist2 = get_dist2_pbc(dat, dat.posx[i], dat.posy[i], dat.posz[i], dat.posx[j], dat.posy[j], dat.posz[j]);
            double pot = get_LJ_pot(dat, dist2);
            energy += pot;
        }
    }

    // energy += dat.N * dat.utail;
    return energy;
}

double get_energy_of(AppData &dat, int k, double deltax, double deltay, double deltaz) {
    double energy = 0;
    for (int i = 0; i < dat.N; i++) {
        if (i == k) continue;

        double dist2 = get_dist2_pbc(dat, dat.posx[i], dat.posy[i], dat.posz[i], dat.posx[k] + deltax, dat.posy[k] + deltay, dat.posz[k] + deltaz);
        double pot = get_LJ_pot(dat, dist2);
        energy += pot;
    }
    // energy += dat.utail;
    return energy;
}

double get_pressure(AppData &dat) {
    double pressure = 0;
    for (int i = 0; i < dat.N; i++) {
        for (int j = 0; j < i; j++) {
            double dist2 = get_dist2_pbc(dat, dat.posx[i], dat.posy[i], dat.posz[i], dat.posx[j], dat.posy[j], dat.posz[j]);
            double ratio = dat.sigma * dat.sigma / dist2;
            pressure += 24 * dat.eps * (pow(ratio, 3) - 2 * pow(ratio, 6));
        }
    }
    pressure += dat.N * dat.ptail;
    pressure = (dat.N * dat.temp - pressure / 3) / (dat.sizex * dat.sizey * dat.sizez);
    return pressure;
}

void assign_random_pos(AppData &dat) {
    for (int i = 0; i < dat.N; i++) {
        dat.posx[i] = random01() * dat.sizex;
        dat.posy[i] = random01() * dat.sizey;
        dat.posz[i] = random01() * dat.sizez;
    }
}

void metropolis_sweep(AppData &dat) {
    for (int i = 0; i < dat.N; i++) {
        if (DEBUG) cout << "Start MC move for " << i << endl;
        double deltax = (random01() - 0.5) * 2 * dat.dmax;
        double deltay = (random01() - 0.5) * 2 * dat.dmax;
        double deltaz = (random01() - 0.5) * 2 * dat.dmax;

        double energy_initial = get_energy_of(dat, i, 0, 0, 0);
        double energy_final = get_energy_of(dat, i, deltax, deltay, deltaz);

        double deltaE = energy_final - energy_initial;
        if (DEBUG) cout << "Proposing a shift of " << deltax << ", " << deltay << ", " << deltaz << endl;
        if (DEBUG) cout << "DeltaE of " << i << " is: " << deltaE << " (" << energy_final << " - " << energy_initial << ")" << endl;

        if (deltaE < 0 || random01() < exp(-deltaE / dat.temp)) {
            if (DEBUG) cout << "Move accepted" << endl;
            dat.posx[i] += deltax;
            dat.posy[i] += deltay;
            dat.posz[i] += deltaz;

            dat.posx[i] = dat.posx[i] - floor(dat.posx[i] / dat.sizex) * dat.sizex;
            dat.posy[i] = dat.posy[i] - floor(dat.posy[i] / dat.sizey) * dat.sizey;
            dat.posz[i] = dat.posz[i] - floor(dat.posz[i] / dat.sizez) * dat.sizez;
        }
        double calcen = get_energy(dat);
        if (DEBUG) cout << "Energy from raw calculation: " << calcen << endl;
    }
}

void save_params_to_file(AppData &dat) {
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.N), sizeof(dat.N));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.niters), sizeof(dat.niters));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.nbuf), sizeof(dat.nbuf));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.save_par_every), sizeof(dat.save_par_every));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.eps), sizeof(dat.eps));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sigma), sizeof(dat.sigma));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.rho), sizeof(dat.rho));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.dmax), sizeof(dat.dmax));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sizex), sizeof(dat.sizex));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sizey), sizeof(dat.sizey));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.sizez), sizeof(dat.sizez));

    dat.stats_file.write(reinterpret_cast<const char *>(&dat.utail), sizeof(dat.utail));
    dat.stats_file.write(reinterpret_cast<const char *>(&dat.ptail), sizeof(dat.ptail));
}

void save_data_to_file(AppData &dat, double *energies, double *pressures) {
    dat.stats_file.write(reinterpret_cast<char *>(energies), dat.nbuf * sizeof(energies[0]));
    dat.stats_file.write(reinterpret_cast<char *>(pressures), dat.nbuf * sizeof(pressures[0]));
}

void save_particle_to_file(AppData &dat) {
    if (!dat.save_particles) return;
    dat.particle_file.write(reinterpret_cast<char *>(dat.posx), dat.N * sizeof(dat.posx[0]));
    dat.particle_file.write(reinterpret_cast<char *>(dat.posy), dat.N * sizeof(dat.posy[0]));
    dat.particle_file.write(reinterpret_cast<char *>(dat.posz), dat.N * sizeof(dat.posz[0]));
}

void run_simulation(AppData &dat) {
    save_params_to_file(dat);
    save_particle_to_file(dat);

    int nbuf = dat.nbuf;
    double *tot_energies = new double[nbuf];
    double *pressures = new double[nbuf];

    tot_energies[0] = get_energy(dat);
    pressures[0] = get_pressure(dat);

    if (DEBUG) cout << "Initial energy: " << tot_energies[0] << endl;

    for (int i = 1; i < dat.niters; i++) {
        metropolis_sweep(dat);

        tot_energies[i % nbuf] = get_energy(dat) + dat.N * dat.utail;
        pressures[i % nbuf] = get_pressure(dat);

        if ((i + 1) % nbuf == 0) {
            save_data_to_file(dat, tot_energies, pressures);
        }

        if (i % dat.save_par_every == 0) {
            save_particle_to_file(dat);
        }
    }
}

void initialize_data(AppData &dat, int N, int niters, double rho, double temp) {
    dat.N = N;
    dat.rho = rho;

    double side = cbrt(dat.N / dat.rho);

    dat.niters = niters;
    dat.nbuf = 1000;
    dat.save_par_every = 1;
    dat.temp = temp;
    dat.eps = 1;
    dat.sigma = 1;
    dat.dmax = 0.5;
    dat.sizex = side;
    dat.sizey = side;
    dat.sizez = side;

    dat.sigma_cut = side / 2;
    dat.utail = 8.0 / 3 * M_PI * dat.rho * (1.0 / 3 * pow(dat.sigma / dat.sigma_cut, 9) - pow(dat.sigma / dat.sigma_cut, 3));
    dat.ptail = 16.0 / 3 * M_PI * dat.rho * dat.rho * (2.0 / 3 * pow(dat.sigma / dat.sigma_cut, 9) - pow(dat.sigma / dat.sigma_cut, 3));
}

int main(int argc, char const *argv[]) {
    AppData dat;

    int N = 300;
    dat.posx = new double[N];
    dat.posy = new double[N];
    dat.posz = new double[N];
    dat.save_particles = false;
    double temp = 0.9;
    double niters = 10000;

    int nrho;
    double rhos[40];
    if (temp == 0.9) {
        double rhos_temp[23] = {0.010468806909828876, 0.045762006347884965, 0.08843852437691835, 0.14603777511837235, 0.2091940267443675, 0.272365887923409, 0.3318070659243455, 0.40795046568499926, 0.5045423799365212, 0.5806857796971749, 0.6606066912950725, 0.7200478692960093, 0.7497372391903843, 0.7621936625214636, 0.7657214215099641, 0.7692023518393258, 0.7780841875227635, 0.7831572922628653, 0.7976273479369376, 0.8026067953587595, 0.8077735574171394, 0.8184192725948278, 0.819761694156824};
        nrho = 23;
        for (int i = 0; i < nrho; i++) rhos[i] = rhos_temp[i];
        
    } else if (temp = 2.0) {
        double rhos_temp[38] = {0.0992636944736629, 0.14262441970946693, 0.16863801770389764, 0.19966465281183832, 0.22168136986441644, 0.25570227849150784, 0.2837210913313426, 0.3007654175172769, 0.340761324433714, 0.38074368260119584, 0.4067437318466711, 0.4337192910169374, 0.4626594187857417, 0.5005823671119977, 0.5284656924622781, 0.5692745243160411, 0.6039322241440378, 0.6317207082516302, 0.6563929800994731, 0.6799949007798367, 0.7115228395991284, 0.7340137628651469, 0.7477250968080447, 0.7603931770813741, 0.7729664161120156, 0.7886558674024067, 0.8011071676924493, 0.8147507578905699, 0.8263213894985093, 0.8465903179358367, 0.8541640686019236, 0.8588519357405033, 0.8674824888251138, 0.8760994931607688, 0.8826841851531091, 0.8883882084633463, 0.89508129044733, 0.901652433690715};
        nrho = 38;
        for (int i = 0; i < nrho; i++) rhos[i] = rhos_temp[i];
    }

    for (int i = 0; i < nrho; i++) {
        double rho = rhos[i];
        initialize_data(dat, N, niters, rho, temp);

        string filename = "LJ_T" + to_string(temp) + "_rho" + to_string(i) + "_N" + to_string(dat.N) + "_M" + to_string(dat.niters) + ".bin";
        dat.stats_file.open(filename, ios::out | ios::binary);

        if (dat.save_particles) {
            dat.particle_file.open("particles.bin", ios::out | ios::binary);
        }

        cout << "Side: " << dat.sizex << endl;
        cout << "u_tail: " << dat.utail << endl;

        assign_random_pos(dat);
        run_simulation(dat);

        dat.stats_file.close();

        if (dat.save_particles) {
            dat.particle_file.close();
        }
    }

    delete[] dat.posx;
    delete[] dat.posy;
    delete[] dat.posz;

    return 0;
}
