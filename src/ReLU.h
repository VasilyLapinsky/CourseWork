#pragma once
#include "Layer.h"

class ReLU : public LayerInterface
{
public:
	Tensor compute(Tensor& input) override;
	std::vector<Tensor> compute(std::vector<Tensor>& input) override;
	Tensor backPropagate(Tensor& input) override;
	std::vector<Tensor> backPropagate(std::vector<Tensor>& input) override;

public:
	void print(std::ostream& out) override;
	Json::Value Serialize() override;
	void DeSerialize(Json::Value json) override;

private:
	Tensor ComputeGradient(Tensor& layerInput, Tensor& gradient);

private:
	Tensor inputs;
	std::vector<Tensor> batchInputs;
};