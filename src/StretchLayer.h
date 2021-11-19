#pragma once
#include "Layer.h"

class StretchLayer : public LayerInterface
{
public:
	StretchLayer(uint width, uint height, uint channels = 1, uint four = 1);

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
	Tensor ReShapeIntoRow(Tensor& input);
	Tensor ReShapeBackward(Tensor& input);

private:
	uint width;
	uint height;
	uint channels;
	uint four;
};