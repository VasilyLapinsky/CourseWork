#pragma once
#include "Common/Tensor.h"

class LayerInterface
{
public:
	virtual ~LayerInterface() = default;

public:
	virtual Tensor compute(Tensor& input) = 0;
	virtual std::vector<Tensor> compute(std::vector<Tensor>& input) = 0;
	virtual Tensor backPropagate(Tensor& input) = 0;
	virtual std::vector<Tensor> backPropagate(std::vector<Tensor>& input) = 0;

public:
	virtual void print(std::ostream& out) = 0;
	virtual void Serialize(Json::Value&, std::ofstream&) = 0;
	virtual void DeSerialize(Json::Value&, std::ifstream&) = 0;
};