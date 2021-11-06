#pragma once
#include "Layer.h"

class SoftMax : public LayerInterface
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
	Tensor ComputeSoftMax(Tensor& input);
	Tensor ComputeError(Tensor& softmaxOutput, Tensor& gradient);

private:
	Tensor output;
	std::vector<Tensor> batchOutputs;
};