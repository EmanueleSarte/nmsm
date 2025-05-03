
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

double get_energy(double J, int nx, int ny, char **spins) {
    double energy = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            energy += spins[i][j] * (spins[i][(j + 1) % ny] + spins[(i + 1) % nx][j]);
        }
    }
    return energy * J * -1;
}

void generate_configuration(int nx, int ny, char **spins) {
    double r;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            r = random01();
            if (r < 0.5) {
                spins[i][j] = -1;
            } else {
                spins[i][j] = 1;
            }
        }
    }
}

int get_magnetization(int nx, int ny, char **spins) {
    int ma = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ma += spins[i][j];
        }
    }
    return ma;
}

void print_configuration(int nx, int ny, char **spins) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (spins[i][j] == 1) {
                std::cout << "+";
            } else {
                std::cout << "-";
            }
        }
        std::cout << std::endl;
    }
}

void save_simulation_data(ofstream &outfile, double energy, int magnet, int nx, int ny, char **spins) {
    outfile.write(reinterpret_cast<const char *>(&energy), sizeof(energy));
    outfile.write(reinterpret_cast<const char *>(&magnet), sizeof(magnet));
    for (int row = 0; row < nx; ++row) {
        outfile.write(reinterpret_cast<const char *>(spins[row]), ny * sizeof(char));
    }
}

void save_parameters(ofstream &outfile, int iters, int nx, int ny, double kB, double T, double J) {
    outfile.write(reinterpret_cast<const char *>(&iters), sizeof(iters));
    outfile.write(reinterpret_cast<const char *>(&nx), sizeof(nx));
    outfile.write(reinterpret_cast<const char *>(&ny), sizeof(ny));
    outfile.write(reinterpret_cast<const char *>(&kB), sizeof(kB));
    outfile.write(reinterpret_cast<const char *>(&T), sizeof(T));
    outfile.write(reinterpret_cast<const char *>(&J), sizeof(J));
}

void run_simulation(string filename, int nsteps, double J, double kB, double T, int nx, int ny, char **spins) {
    ofstream outfile(filename, ios::out | ios::binary);
    save_parameters(outfile, nsteps, nx, ny, kB, T, J);

    double energy = get_energy(J, nx, ny, spins);
    int magnet = get_magnetization(nx, ny, spins);

    save_simulation_data(outfile, energy, magnet, nx, ny, spins);

    double exp_values[] = {exp(8 * J / (kB * T)), exp(4 * J / (kB * T)), exp(0 * J / (kB * T)), exp(-4 * J / (kB * T)), exp(-8 * J / (kB * T))};

    for (int m = 0; m < nsteps; m++) {
        // for (int k = 0; k < nx * ny; k++) {
        //     int i = (int)(random01() * nx);
        //     int j = (int)(random01() * ny);

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                char neigh_sum = spins[i][(j + 1) % ny] + spins[i][(j - 1 + ny) % ny] + spins[(i + 1) % nx][j] + spins[(i - 1 + nx) % nx][j];
                char offset = neigh_sum * spins[i][j];
                char deltaE = offset * 2;

                if (deltaE < 0 || (random01() < exp_values[(offset + 4) / 2])) {
                    energy += deltaE;
                    magnet += -2 * spins[i][j];

                    spins[i][j] = spins[i][j] * -1;
                }
            }
        }

        save_simulation_data(outfile, energy, magnet, nx, ny, spins);
    }
    outfile.close();
}

int main(int argc, const char *argv[]) {
    std::string filename = "";
    int nsteps = -1;
    double J = -1.0;
    double kB = -1.0;
    double T = -1.0;
    int nx = -1;
    int ny = -1;

    // int nsteps = 4000;
    // double J = 1;
    // double kB = 1;
    // double T = 3;
    // int nx = 100;
    // int ny = 100;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            int eqPos = arg.find('=');
            if (eqPos == std::string::npos) {
                throw std::runtime_error("Invalid parameter format: " + arg + ". Use name=value.");
            }

            std::string name = arg.substr(0, eqPos);
            std::string valueStr = arg.substr(eqPos + 1);

            if (name == "filename") {
                filename = valueStr;
            } else if (name == "nsteps") {
                nsteps = std::stoi(valueStr);
            } else if (name == "J" || name == "j") {
                J = std::stod(valueStr);
            } else if (name == "kB" || name == "kb") {
                kB = std::stod(valueStr);
            } else if (name == "T" || name == "t") {
                T = std::stod(valueStr);
            } else if (name == "nx") {
                nx = std::stoi(valueStr);
            } else if (name == "ny") {
                ny = std::stoi(valueStr);
            } else {
                std::cerr << "Warning: Unknown parameter '" << name << "' will be ignored." << std::endl;
            }
        }

        if (filename.empty() || nsteps == -1 || J == -1.0 || kB == -1.0 || T == -1.0 || nx == -1 || ny == -1) {
            throw std::runtime_error("Missing required parameters. Please specify filename, nsteps, J, kB, T, nx, and ny.");
        }

        std::cout << "Filename: " << filename << std::endl;
        std::cout << "Number of steps: " << nsteps << std::endl;
        std::cout << "J: " << J << std::endl;
        std::cout << "kB: " << kB << std::endl;
        std::cout << "T: " << T << std::endl;
        std::cout << "nx: " << nx << std::endl;
        std::cout << "ny: " << ny << std::endl;

    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Usage: program_name filename=<string> nsteps=<int> J=<double> kB=<double> T=<double> nx=<int> ny=<int>" << std::endl;
        return 1;
    } catch (const std::invalid_argument &e) {
        std::cerr << "Error: Invalid argument: " << e.what() << std::endl;
        std::cerr << "Usage: program_name filename=<string> nsteps=<int> J=<double> kB=<double> T=<double> nx=<int> ny=<int>" << std::endl;
        return 1;
    } catch (const std::out_of_range &e) {
        std::cerr << "Error: Value out of range: " << e.what() << std::endl;
        std::cerr << "Usage: program_name filename=<string> nsteps=<int> J=<double> kB=<double> T=<double> nx=<int> ny=<int>" << std::endl;
        return 1;
    }

    std::cout.precision(10);

    char *memory = new char[nx * ny];
    char **spins = new char *[nx];

    for (int i = 0; i < nx; i++) {
        spins[i] = &memory[i * ny];
    }

    generate_configuration(nx, ny, spins);
    run_simulation(filename, nsteps, J, kB, T, nx, ny, spins);

    delete[] memory;
    delete[] spins;

    return 0;
}
