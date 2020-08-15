#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

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
