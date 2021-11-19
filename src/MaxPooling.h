#pragma once
#include "Layer.h"

class MaxPooling : public LayerInterface
{
public:
	MaxPooling(uint pool, uint stride);

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
	Tensor ComputeGradient(Tensor& layerInput, Tensor& gradient);

public:
	uint pool;
	uint stride;

	Tensor inputs;
	std::vector<Tensor> batchInputs;
};