#pragma once
#include "NeuralNet/Layer.h"

class ConvLayer : public LayerInterface
{
public:
	ConvLayer(double lambda, uint kernelSize, uint inputChannels, 
			uint numFilters, uint stride, uint padding=0);

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
	Tensor ConvolveTensor(Tensor& value);

public:
	double lambda;
	uint stride;
	uint padding;
	uint kernelSize;

	Tensor weights;
	Tensor bias;
	Tensor dwMemmory;

	Tensor inputs;
};