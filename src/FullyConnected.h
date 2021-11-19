#pragma once
#include "Layer.h"

class FullyConnected : public LayerInterface
{
public:
	FullyConnected(uint numInputs, uint numOutput, double lambda);

public:
	Tensor compute(Tensor& input) override;
	std::vector<Tensor> compute(std::vector<Tensor>& input) override;
	Tensor backPropagate(Tensor& input) override;
	std::vector<Tensor> backPropagate(std::vector<Tensor>& input) override;

public:
	void print(std::ostream&) override;
	void Serialize(Json::Value& config, std::ofstream& weigths) override;
	void DeSerialize(Json::Value& config, std::ifstream& weigths) override;
private:
	Tensor Forward(Tensor& input);
	std::tuple<Tensor, Tensor, Tensor> ComputeGradient(Tensor& layerInput, Tensor& gradient);

public:
	double lambda;

	Tensor weights;
	Tensor bias;
	Tensor inputs;
	std::vector<Tensor> batchInputs;
};