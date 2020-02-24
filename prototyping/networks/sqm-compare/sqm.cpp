// hard-coded for molecule 12 in 10.1063/1.5088083 
// g++ -O4 -std=c++17 -fopenmp prescan.cpp
#include<iostream>
#include<fstream>
#include<vector>
#include<cstdint>
#include<sstream>
#include<ctime>
#include <string.h>
#include"xtb/xtb.h"

int main(int argc, char ** argv) {
	// read raw binary graphs
	std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	file.read(buffer.data(), size);
	int nentries = buffer.size()/22;
	uint8_t * nums = reinterpret_cast<uint8_t *>(buffer.data());
	std::cout << "# loaded " << nentries << std::endl;
	
	// prepare meta data
	const int natoms = 36;
	int attyp[36];
	std::fill_n(attyp, 36, 1);
	const double charge = 0;
	size_t pos;
	double coord[36*3] = { 2.32917700,  3.43079900,  0.00000000,  2.29549200,  2.04802500, -0.00000000,  1.06185900,  1.34803700, -0.00000000, -0.16868200,  2.07625300,  0.00000000, -0.09780500,  3.49363400,  0.00000000,  1.11887100,  4.15653400,  0.00000000, -1.42713200,  1.34598000, -0.00000000, -2.68188500,  2.01851200, -0.00000000, -3.87118900,  1.33190300, -0.00000000, -3.91620500, -0.09104900, -0.00000000, -5.17785200, -0.81582700, -0.00000000, -6.44074000, -0.16848100, -0.00000000, -7.62319500, -0.89071700, -0.00000000, -7.59981800, -2.30176200,  0.00000000, -6.38548800, -2.96401200,  0.00000000, -5.16248100, -2.24560600,  0.00000000, -3.90399900, -2.92521900,  0.00000000, -2.72244800, -2.23637000,  0.00000000, -2.68295900, -0.80213200, -0.00000000, -1.42799700, -0.07756000, -0.00000000, -0.16618600, -0.76048100, -0.00000000,  1.02116100, -0.08166600, -0.00000000,  3.28505400,  3.95993400,  0.00000000,  3.22434200,  1.47133200, -0.00000000, -1.01449200,  4.08447100,  0.00000000,  1.13930900,  5.24900500,  0.00000000, -2.71077500,  3.10832800, -0.00000000, -4.80046200,  1.90196300, -0.00000000, -6.49391700,  0.92082200, -0.00000000, -8.57955400, -0.36223900, -0.00000000, -8.53602000, -2.86498900,  0.00000000, -6.35044000, -4.05676800,  0.00000000, -3.90056500, -4.01834900,  0.00000000, -1.79060900, -2.80117900,  0.00000000, -0.14307500, -1.84988800, -0.00000000,  1.96610900, -0.63121900, -0.00000000, };
	xtb::SCC_options opt;
	std::string solvent = "none";
	strcpy(opt.solvent, solvent.c_str());
	opt.etemp = 300;
	opt.acc = 1;
	opt.maxiter = 250;
	opt.parallel = 1;
	opt.grad = false;
	double energy;
	double dipole[3];
	double q[36];
	double qp[6*36];
	double wbo[36*36];
	const double * coords = const_cast<double*>(&coord[0]);
	for (int i = 0; i < 36*3; i++) {
		coord[i] *= 1.88973;
	}

	// arguments
	size_t begin = atoi(argv[2]);
	size_t end = atoi(argv[3]);
	std::time_t begintime = std::time(nullptr);

	// loop over molecules
	//#pragma omp parallel for
	for (size_t i = begin; i < end; i++) {
		std::stringstream stream;
		
		for (pos = 0; pos < 22; pos ++){
			attyp[pos] = nums[i*22 + pos];
			/*if (attyp[pos] == 8) {
				attyp[pos]=5;
			}*/
		}
		GFN2_calculation(&natoms, attyp, &charge, 0, coords, &opt, "/dev/null", &energy, NULL, dipole, q, NULL, qp, wbo);
		stream << "energy" << energy << std::endl;
		std::cout << stream.str();
	}
	std::time_t endtime = std::time(nullptr);
	double mcmps = (end+begin)/2;
	mcmps = mcmps * (nentries - mcmps);
	mcmps /= std::difftime(endtime, begintime) * 1000000;

	// finalize
	std::cout << "# done, " << end-begin << " molecules in " << std::difftime(endtime, begintime) << "s, " << mcmps << " Mcmp/s" << std::endl;
}
