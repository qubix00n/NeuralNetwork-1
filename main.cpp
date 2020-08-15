#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

#define STATS 1
#define LEARNING_RATE 0.25f

template<class T> std::ostream& operator << (std::ostream& out, const std::vector<T>& v) {
	// out << "{ "; for (int i = 0; i < v.size() - 1; i++) out << v[i] << ", "; out << v[v.size() - 1] << " }"; return out;
	for (int i = 0; i < v.size(); i++) out << v[i] << ' '; out << '\n';  return out;
}

float tanhDerivative(float v) {
	return 1 - (v * v);
}

#include "Layer.cpp"
#include "NeuralNetwork.cpp"

int main() {
	srand(time(NULL));

	std::cout << "started with"; if (!STATS) std::cout << "out"; std::cout << " statistics\n";

	NeuralNetwork net({ 3, 5, 1 });

	{
		int iterations = 50000, progressStep = iterations / 50;

		std::cout << "started training for " << iterations << " iterations" << std::endl;

		auto train = [&](std::vector<float> inputs, std::vector<float> expected) {
			net.feedForward(inputs); net.backProp(expected);
		};

#if STATS
		std::ofstream file; file.open("error.csv");
#endif

		for (int i = 0; i < iterations + 1; i++) if (!(i % progressStep)) std::cout << '#'; std::cout << "<\n";
		for (int i = 0; i < iterations + 1; i++) {
			train({ 0, 0, 1 }, { 0 });
			train({ 0, 1, 0 }, { 0 });
			train({ 0, 1, 1 }, { 0 });
			train({ 1, 0, 0 }, { 1 });
			train({ 1, 0, 1 }, { 0 });
			train({ 1, 1, 0 }, { 1 });
			train({ 1, 1, 1 }, { 1 });
			if (!(i % progressStep)) std::cout << "#";
#if STATS
			if (!(i % 100)) file << i
				<< ',' << abs(0 - net.feedForward({ 0, 0, 1 })[0])
				<< ',' << abs(0 - net.feedForward({ 0, 1, 0 })[0])
				<< ',' << abs(0 - net.feedForward({ 0, 1, 1 })[0])
				<< ',' << abs(1 - net.feedForward({ 1, 0, 0 })[0])
				<< ',' << abs(0 - net.feedForward({ 1, 0, 1 })[0])
				<< ',' << abs(1 - net.feedForward({ 1, 1, 0 })[0])
				<< ',' << abs(1 - net.feedForward({ 1, 1, 1 })[0]) << '\n';
#endif
		}

#if STATS
		file.close();
#endif

		std::cout << "<\ntraining finished\n";
	}

	/*const char* stateFilename = "state.txt";
	std::ofstream state; state.open(stateFilename);
	state << net; std::cout << "NN state saved as: " << stateFilename << '\n';
	state.close();*/

	/*const char* stateFilename = "state.txt";
	std::ifstream _state(stateFilename);
	NeuralNetwork net_recovered({ 3, 5, 1 }, _state);
	_state.close();*/

	std::cout << "\n\n";

	return 177013;
}
