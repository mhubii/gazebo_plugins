#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "gazebo_utils.h"

// First argument takes number of samples, second takes center x,
// third takes center y, and the fourth takes the radius r
// e.g. ./sample 1000 0.5 0.5 0.5
int main(int argc, char** argv) {


	std::ofstream out;
	out.open("circular_random_sample.csv");

	int samples = atoi(argv[1]);
	float x = atof(argv[2]);
	float y = atof(argv[3]);
	float r = atof(argv[4]);
	
	for (int i = 0; i < samples; i++) {

		Eigen::Vector2d rv = UniformCircularRandVar(x, y, r);

		out << rv(0) << ", " << rv(1) << "\n";
	}

	out.close();

	return 0;
}
