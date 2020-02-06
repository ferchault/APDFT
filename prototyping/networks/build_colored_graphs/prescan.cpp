#include<iostream>
#include<fstream>
#include<vector>
#include<cstdint>
#include<sstream>
#include<ctime>

bool group_precheck(uint8_t * a, uint8_t * b) {
	// [[0, 5, 12, 13], [1, 14], [2, 15], [3, 10], [4, 11], [6, 9, 18, 19], [7, 8, 17, 20], [16, 21]]
	if (a[1] - b[1] + a[14] - b[14]) { return false; }
	if (a[2] - b[2] + a[15] - b[15]) { return false; }
	if (a[3] - b[3] + a[10] - b[10]) { return false; }
	if (a[4] - b[4] + a[11] - b[11]) { return false; }
	if (a[16] - b[16] + a[21] - b[21]) { return false; }
	if (a[0] - b[0] + a[5] - b[5] + a[12] - b[12] + a[13] - b[13]) { return false; }
	if (a[6] - b[6] + a[9] - b[9] + a[18] - b[18] + a[19] - b[19]) { return false; }
	// last one not needed, implicitly checked
	return true;
}
bool is_even(uint8_t * a, uint8_t * b) {
	int count = 0;
	for (int i =0; i < 22; i++) {
		if (a[i] != b[i]) {
			count += 1;
		}
	}
	return (count % 2) == 0;
}

int main(int argc, char ** argv) {
	// read raw binary graphs
	std::ifstream file("bin", std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	file.read(buffer.data(), size);
	int nentries = buffer.size()/22;
	uint8_t * nums = reinterpret_cast<uint8_t *>(buffer.data());
	std::cout << "# loaded " << nentries << std::endl;
	
	// arguments
	size_t begin = atoi(argv[1]);
	size_t end = atoi(argv[2]);
	std::time_t begintime = std::time(nullptr);

	// loop over molecules
	#pragma omp parallel for
	for (size_t i = begin; i < end; i++) {
		std::stringstream stream;
		// loop over other molecules
		for (int j = i+1; j < nentries; j++){
			if (not group_precheck(nums + i*22, nums + j*22)) { continue; }
			if (not is_even(nums + i*22, nums + j*22)) { continue; }
			stream << i << " " << j << std::endl;
		}
		std::cout << stream.str();
	}
	std::time_t endtime = std::time(nullptr);
	double mcmps = (end+begin)/2;
	mcmps = mcmps * (nentries - mcmps);
	mcmps /= (endtime - begintime) * 1000000;

	// finalize
	std::cout << "# done, " << end-begin << " molecules in " << endtime - begintime << "s, " << mcmps << " Mcmp/s" << std::endl;
}
