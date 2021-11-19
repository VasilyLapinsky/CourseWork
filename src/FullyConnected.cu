#include "FullyConnected.h"
#include "MathFunctions.h"
#include <iostream>

const char* FullyConnectedConfigNodeName = "FullyConnected";

FullyConnected::FullyConnected(uint numInputs, uint numOutput, double lambda)
	: lambda{lambda}
	, weights(GenerateUniformDistributionTensor(numOutput, numInputs, 1, 1) * 0.01)
	, bias(GenerateFilledTensor(numOutput, 1, 1, 1, 0))
{
}

Tensor FullyConnected::compute(Tensor& input)
{
	this->inputs = input;
	return this->Forward(this->inputs);
}

std::vector<Tensor> FullyConnected::compute(std::vector<Tensor>& input)
{
	this->batchInputs = input;

	std::vector<Tensor> result;
	std::transform(input.begin(), input.end(), std::back_inserter(result),
		[this](Tensor& value) { return this->Forward(value); });

	return result;
}

Tensor FullyConnected::backPropagate(Tensor& input)
{
	auto gradients = this->ComputeGradient(this->inputs, input);

	auto dx = std::get<0>(gradients);
	auto dw = std::get<1>(gradients);
	auto db = std::get<2>(gradients);

	this->weights -=  TransposeMatrix(dw, 0) * this->lambda;
	this->bias -= TransposeMatrix(db, 0) * this->lambda;

	return dx;
}

std::vector<Tensor> FullyConnected::backPropagate(std::vector<Tensor>& input)
{
	if (input.size() != this->batchInputs.size())
	{
		throw std::invalid_argument("Batches must be same size!");
	}

	std::vector<Tensor> result(input.size());
	Tensor weightsGradient = 
		GenerateFilledTensor(this->weights.height, this->weights.width, this->weights.channels, this->weights.four, 0);

	Tensor biasGradient =
		GenerateFilledTensor(this->bias.height, this->bias.width, this->bias.channels, this->bias.four, 0);

	for (int i = 0; i < input.size(); ++i)
	{
		auto gradients = this->ComputeGradient(this->batchInputs[i], input[i]);

		auto dx = std::get<0>(gradients);
		auto dw = std::get<1>(gradients);
		auto db = std::get<2>(gradients);

		result[i] = dx;

		weightsGradient += dw;
		biasGradient += db;
	}

	weightsGradient *= 1. / static_cast<double>(input.size());
	biasGradient *= 1. / static_cast<double>(input.size());

	this->weights -= TransposeMatrix(weightsGradient, 0) * this->lambda;
	this->bias -= TransposeMatrix(biasGradient, 0) * this->lambda;

	return result;
}

void FullyConnected::print(std::ostream& out)
{
	out << "FullyConnected\n";
	out << "lambda: " << this->lambda << '\n';
	out << "Weights: " << TensorToCvMat(this->weights) << '\n';
	out << "Bias: " << TensorToCvMat(this->bias) << '\n';
}

void FullyConnected::Serialize(Json::Value& config, std::ofstream& weigths)
{
	config[FullyConnectedConfigNodeName]["lambda"] = this->lambda;
	this->weights.Serrialize(config[FullyConnectedConfigNodeName]["weights"], weigths);
	this->bias.Serrialize(config[FullyConnectedConfigNodeName]["bias"], weigths);
}

void FullyConnected::DeSerialize(Json::Value& config, std::ifstream& weigths)
{
	this->lambda = config[FullyConnectedConfigNodeName]["lambda"].asDouble();
	this->weights = Tensor(config[FullyConnectedConfigNodeName]["weights"], weigths);
	this->bias = Tensor(config[FullyConnectedConfigNodeName]["bias"], weigths);
}

Tensor FullyConnected::Forward(Tensor& input)
{
	return MatrixMult(input, 0, this->weights, 0) + this->bias;
}

std::tuple<Tensor, Tensor, Tensor> FullyConnected::ComputeGradient(Tensor& layerInput, Tensor& gradient)
{
	Tensor dy = gradient;
	if (layerInput.height == dy.height)
	{
		dy = TransposeMatrix(dy, 0);
	}

	Tensor dw = MatrixMult(dy, 0, layerInput, 0);
	Tensor db = ReduceCols(dy);

	Tensor dx = MatrixMult(TransposeMatrix(dy, 0), 0, TransposeMatrix(this->weights, 0), 0);

	return { dx, dw, db };
}