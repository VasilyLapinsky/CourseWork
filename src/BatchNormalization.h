#pragma once
#include "Layer.h"

class BatchNormalization : public LayerInterface
{
public:
	BatchNormalization(double lambda, uint width, uint height, uint channels = 1, uint four = 1);

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
	void Standardization(std::vector<Tensor>& input);
	Tensor Scale(Tensor& input);

	Tensor ComputeGradient(Tensor& dldy);

private:
	double lambda;

	Tensor gamma;
	Tensor beta;

	std::vector<Tensor> batchInputs;
	std::vector<Tensor> standartized;
	Tensor mean;
	Tensor variance;
};