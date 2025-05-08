
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

struct AppParams {
    double a;
    double b;
    int omega;

    int niters;
    int s0_0;
    int s0_1;

    int nbuf;

    ofstream stats_file;
};

double random01() {
    static std::random_device rd;                           // Seed source
    static std::mt19937 gen(rd());                          // Mersenne Twister RNG
    static std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform in [0, 1)
    return dis(gen);
    // static double randomNumbers[100] = {0.5434049417909654, 0.27836938509379616, 0.4245175907491331, 0.8447761323199037, 0.004718856190972565, 0.12156912078311422, 0.6707490847267786, 0.8258527551050476, 0.13670658968495297, 0.57509332942725, 0.891321954312264, 0.20920212211718958, 0.18532821955007506, 0.10837689046425514, 0.21969749262499216, 0.9786237847073697, 0.8116831490893233, 0.1719410127325942, 0.8162247487258399, 0.2740737470416992, 0.4317041836631217, 0.9400298196223746, 0.8176493787767274, 0.3361119501208987, 0.17541045374233666, 0.37283204628992317, 0.005688507352573424, 0.25242635344484043, 0.7956625084732873, 0.01525497124633901, 0.5988433769284929, 0.6038045390428536, 0.10514768541205632, 0.38194344494311006, 0.03647605659256892, 0.8904115634420757, 0.9809208570123115, 0.05994198881803725, 0.8905459447285041, 0.5769014994000329, 0.7424796890979773, 0.6301839364753761, 0.5818421923987779, 0.020439132026923157, 0.2100265776728606, 0.5446848781786475, 0.7691151711056516, 0.2506952291383959, 0.2858956904068647, 0.8523950878413064, 0.9750064936065875, 0.8848532934911055, 0.35950784393690227, 0.5988589458757472, 0.3547956116572998, 0.34019021537064575, 0.17808098950580487, 0.23769420862405044, 0.04486228246077528, 0.5054314296357892, 0.376252454297363, 0.5928054009758866, 0.6299418755874974, 0.14260031444628352, 0.933841299466419, 0.9463798808091013, 0.6022966577308656, 0.38776628032663074, 0.3631880041093498, 0.20434527686864423, 0.27676506139633517, 0.24653588120354963, 0.17360800174020508, 0.9666096944873236, 0.9570126003527981, 0.5979736843289207, 0.7313007530599226, 0.3403852228374361, 0.09205560337723862, 0.4634980189371477, 0.508698893238194, 0.08846017300289077, 0.5280352233180474, 0.9921580365105283, 0.3950359317582296, 0.3355964417185683, 0.8054505373292797, 0.7543489945823536, 0.3130664415885097, 0.6340366829622751, 0.5404045753007164, 0.2967937508800147, 0.11078790118244575, 0.3126402978757431, 0.4569791300492658, 0.6589400702261969, 0.2542575178177181, 0.6411012587007017, 0.20012360721840317, 0.6576248055289837};
    // static int index = -1;
    // index += 1;

    // return randomNumbers[index];
}


double get_tau(double *rates, double sumr) {
    double r = random01();
    return log(1.0 / r) / sumr;
}

void get_rates(AppParams &par, int *state, double *rates) {
    rates[1] = state[0];
    // the order of op per rates[2] is to avoid int32 overflow
    rates[2] = (state[0] / pow(par.omega, 2)) * state[1] * (state[0] - 1);
    rates[3] = par.b * state[0];
}

int select_reaction(double *rates, double sumr) {
    double r = random01();
    double foo = sumr * r;
    double bar = 0;
    for (int i = 0; i < 4; i++) {
        bar += rates[i];
        if (bar > foo) return i;
    }
    return -1;
}

void update_state(int *state, int reaction) {
    switch (reaction) {
        case 0:
            state[0] += 1;
            break;
        case 1:
            state[0] -= 1;
            break;
        case 2:
            state[0] += 1;
            state[1] -= 1;
            break;
        case 3:
            state[0] -= 1;
            state[1] += 1;
            break;
        default:
            break;
    }
}

void save_params_to_file(AppParams &par) {
    par.stats_file.write(reinterpret_cast<const char *>(&par.niters), sizeof(par.niters));
    par.stats_file.write(reinterpret_cast<const char *>(&par.a), sizeof(par.a));
    par.stats_file.write(reinterpret_cast<const char *>(&par.b), sizeof(par.b));
    par.stats_file.write(reinterpret_cast<const char *>(&par.omega), sizeof(par.omega));
    par.stats_file.write(reinterpret_cast<const char *>(&par.s0_0), sizeof(par.s0_0));
    par.stats_file.write(reinterpret_cast<const char *>(&par.s0_1), sizeof(par.s0_1));
    par.stats_file.write(reinterpret_cast<const char *>(&par.nbuf), sizeof(par.nbuf));
}

void save_data_to_file(AppParams &par, double *times, int *states0, int *states1) {  // Write the number of elements
    par.stats_file.write(reinterpret_cast<char *>(times), par.nbuf * sizeof(times[0]));
    par.stats_file.write(reinterpret_cast<char *>(states0), par.nbuf * sizeof(states0[0]));
    par.stats_file.write(reinterpret_cast<char *>(states1), par.nbuf * sizeof(states1[0]));
}

void run_simulation(AppParams &par) {
    save_params_to_file(par);

    int nbuf = par.nbuf;
    double *times = new double[nbuf];
    int *states0 = new int[nbuf];
    int *states1 = new int[nbuf];

    double rates[4] = {par.a * par.omega, 0, 0, 0};
    int state[2] = {par.s0_0, par.s0_1};
    double sumr;
    double t = 0;

    for (int i = 0; i < par.niters; i++) {
        get_rates(par, state, rates);
        sumr = rates[0] + rates[1] + rates[2] + rates[3];
        double tau = get_tau(rates, sumr);
        int k = select_reaction(rates, sumr);
        update_state(state, k);
        t += tau;

        states0[i % nbuf] = state[0];
        states1[i % nbuf] = state[1];
        times[i % nbuf] = t;

        if ((i+1) % nbuf == 0) {
            save_data_to_file(par, times, states0, states1);
        }
    }

    delete[] times;
    delete[] states0;
    delete[] states1;
}

int main(int argc, char const *argv[]) {
    AppParams par;
    par.a = 2;
    par.b = 5;
    par.omega = 10000;
    par.niters = 20000000;
    par.s0_0 = par.omega * 2.5;
    par.s0_1 = par.omega * 2.5;
    par.nbuf = 10000;

    string filename = "omega" + to_string(par.omega) + "_iter" + to_string(par.niters) + ".bin";
    par.stats_file.open(filename, ios::out | ios::binary);

    run_simulation(par);

    par.stats_file.close();

    return 0;
}
