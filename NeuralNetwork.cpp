#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

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
