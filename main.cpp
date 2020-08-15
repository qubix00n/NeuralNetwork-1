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

struct Layer {
	int numOfInputs_;
	int numOfOutputs_;

	std::vector<float> inputs_;
	std::vector<float> outputs_;
	std::vector<std::vector<float>> weights_;
	std::vector<std::vector<float>> weightsDelta_;
	std::vector<float> gamma_;
	std::vector<float> error_;

	void initializeWeights() {
		weights_.clear();
		for (int i = 0; i < numOfOutputs_; i++) {
			std::vector<float> temp;
			for (int j = 0; j < numOfInputs_; j++)
				temp.push_back((rand() % 10000) / 10000.0f - 0.5f);
			weights_.push_back(temp);
		}
	}

	void updateWeights() {
		for (int i = 0; i < numOfOutputs_; i++)
			for (int j = 0; j < numOfInputs_; j++)
				weights_[i][j] -= weightsDelta_[i][j] * LEARNING_RATE;
	}

	void backPropOutput(std::vector<float> expected) {
		error_.clear();
		gamma_.clear();
		weightsDelta_.clear();
		for (int i = 0; i < numOfOutputs_; i++)
			error_.push_back(outputs_[i] - expected[i]);
		for (int i = 0; i < numOfOutputs_; i++)
			gamma_.push_back(error_[i] * tanhDerivative(outputs_[i]));
		for (int i = 0; i < numOfOutputs_; i++) {
			std::vector<float> temp;
			for (int j = 0; j < numOfInputs_; j++)
				temp.push_back(gamma_[i] * inputs_[j]);
			weightsDelta_.push_back(temp);
		}
	}

	void backPropHidden(std::vector<float> gammaForward, std::vector<std::vector<float>> weightsForward) {
		gamma_.clear();
		for (int i = 0; i < numOfOutputs_; i++) {
			gamma_.push_back(0);
			for (int j = 0; j < gammaForward.size(); j++)
				gamma_[i] += gammaForward[j] * weightsForward[j][i];
			gamma_[i] *= tanhDerivative(outputs_[i]);
		}
		weightsDelta_.clear();
		for (int i = 0; i < numOfOutputs_; i++) {
			std::vector<float> temp;
			for (int j = 0; j < numOfInputs_; j++)
				temp.push_back(gamma_[i] * inputs_[j]);
			weightsDelta_.push_back(temp);
		}
	}

	Layer(int numOfInputs, int numOfOutputs) : numOfInputs_(numOfInputs), numOfOutputs_(numOfOutputs) {
		inputs_.reserve(numOfInputs);
		outputs_.reserve(numOfOutputs);
		initializeWeights();
	}

	std::vector<float> feedForward(std::vector<float> inputs) {
		inputs_ = inputs;
		outputs_.clear(); for (int i = 0; i < numOfOutputs_; i++) outputs_.push_back(0);
		for (int i = 0; i < numOfOutputs_; i++) {
			outputs_[i] = 0;
			for (int j = 0; j < numOfInputs_; j++) {
				outputs_[i] += inputs_[j] * weights_[i][j];
			}
			outputs_[i] = tanh(outputs_[i]);
		}
		return outputs_;
	}
};

struct NeuralNetwork {
	std::vector<int> layer_;
	std::vector<Layer> layers_;

	NeuralNetwork(std::vector<int> layer) : layer_(layer) {
		for (int i = 0; i < layer.size() - 1; i++)
			layers_.push_back(Layer(layer_[i], layer_[i + 1]));
	}

	NeuralNetwork(std::vector<int> layer, std::ifstream& state) : layer_(layer) {
		for (int i = 0; i < layer.size() - 1; i++)
			layers_.push_back(Layer(layer_[i], layer_[i + 1]));

		std::vector<float> states; std::string str; while (getline(state, str)) {
			float f = std::stof(str);
			states.push_back(f);
		}

		int i = 0; for (Layer& l : layers_) {
			for (std::vector<float>& v : l.weights_)
				for (float& f : v)
					f = states[i++];
		}

		std::cout << "sucesfully recreated a NN from file\n";
	}

	std::vector<float> feedForward(std::vector<float> inputs) {
		layers_[0].feedForward(inputs);
		for (int i = 1; i < layers_.size(); i++) {
			layers_[i].feedForward(layers_[i - 1].outputs_);
		}
		return layers_[layers_.size() - 1].outputs_;
	}

	void backProp(std::vector<float> expected) {
		for (int i = layers_.size() - 1; i >= 0; i--)
			if (i == layers_.size() - 1)
				layers_[i].backPropOutput(expected);
			else
				layers_[i].backPropHidden(layers_[i + 1].gamma_, layers_[i + 1].weights_);
		for (int i = 0; i < layers_.size(); i++)
			layers_[i].updateWeights();
	}
};

std::ostream& operator << (std::ostream& out, const NeuralNetwork& nn) {
	for (const Layer &l : nn.layers_)
		for (const std::vector<float>& v : l.weights_)
			for (const float &f : v)
				out << f << '\n';
	return out;
}

int main() {
	srand(time(NULL));

	std::cout << "started with"; if (!STATS) std::cout << "out"; std::cout << " statistics\n";

	NeuralNetwork net({ 3, 5, 1 });

	// training
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

	const char* stateFilename = "state.txt";
	
	std::ofstream state; state.open(stateFilename);
	state << net; std::cout << "NN state saved as: " << stateFilename << '\n';
	state.close();

	std::ifstream _state(stateFilename);
	NeuralNetwork net_recovered({ 3, 5, 1 }, _state);
	_state.close();

	std::cout << "\n\n\n\n\n";

	return 177013;
}
