#pragma once
#include "NeuralNet/Layer.h"

class ReLU : public LayerInterface
{
public:
	Tensor compute(Tensor& input) override;
	std::vector<Tensor> compute(std::vector<Tensor>& input) override;
	Tensor backPropagate(Tensor& input) override;
	std::vector<Tensor> backPropagate(std::vector<Tensor>& input) override;

public:
	void print(std::ostream& out) override;
	void Serialize(Json::Value& config, std::ofstream& weigths) override;
	void DeSerialize(Json::Value& config, std::ifstream& weigths) override;

private:
	Tensor ComputeGradient(Tensor& layerInput, Tensor& gradient);

private:
	Tensor inputs;
	std::vector<Tensor> batchInputs;
};