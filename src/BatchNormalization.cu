#include "BatchNormalization.h"
#include "HelpTools.h"
#include "MathFunctions.h"
#include <numeric>
#include <iostream>

const uint MAX_THREADS = 32;
const double EPS = 1e-6;

BatchNormalization::BatchNormalization(double lambda, uint width, uint height, uint channels, uint four)
	: lambda{lambda}
	, beta{ GenerateFilledTensor(width, height, channels, four, 0) }
	, gamma{ GenerateFilledTensor(width, height, channels, four, 1) }
{
}


Tensor BatchNormalization::compute(Tensor& input)
{
	return this->Scale(input);
}

std::vector<Tensor> BatchNormalization::compute(std::vector<Tensor>& input)
{
	this->batchInputs = input;
	
	this->Standardization(input);

	std::vector<Tensor> result;
	std::transform(this->standartized.begin(), this->standartized.end(), std::back_inserter(result),
		[this](Tensor& value) {return this->Scale(value); });

	return result;
}

Tensor BatchNormalization::backPropagate(Tensor& input)
{
	return input;
}

__global__ void ComputeVarianceGradient(double* inputs, double* gradStandartized, 
					double* mean, double* sigmas, double* gradient, 
					uint gradientSize, uint batchSize, double eps) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < gradientSize) {

		for (uint i = 0; i < batchSize; ++i)
		{
			gradient[tid] += gradStandartized[i*gradientSize + tid] * (inputs[i * gradientSize + tid] - mean[tid]);
		}

		double sigma = sigmas[tid] + eps;
		gradient[tid] *= - 0.5 / (sigma * sqrt(sigma));

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void ComputeMeanGradient(double* inputs, double* gradStandartized,
	double* mean, double* varience, double* gradientVariance, double* gradient,
	uint gradientSize, uint batchSize, double eps) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < gradientSize) {

		double leftPart = 0;
		double rightPart = 0;

		for (uint i = 0; i < batchSize; ++i)
		{
			leftPart += gradStandartized[i * gradientSize + tid];
			rightPart += inputs[i * gradientSize + tid] - mean[tid];
		}

		leftPart /= -sqrt(varience[tid] + eps);
		rightPart *= -2.0 * gradientVariance[tid] / static_cast<double>(batchSize);
		gradient[tid] = leftPart + rightPart;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void ComputeOutputGradient(double* inputs, double* gradStandartized,
									double* mean, double* gradientMean,
									double* varience, double* gradientVariance, double* gradient,
									uint inputSize, uint batchSize, double eps) {

	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < inputSize) {

		for (uint i = 0; i < batchSize; ++i)
		{
			gradient[i * inputSize + tid] = gradStandartized[i * inputSize + tid] / sqrt(varience[tid] + eps);
			gradient[i * inputSize + tid] += 2.0 * gradientVariance[tid] * 
											(inputs[i * inputSize + tid] - mean[tid]) / static_cast<double>(batchSize);
			gradient[i * inputSize + tid] += gradientMean[tid] / static_cast<double>(batchSize);
		}

		tid += blockDim.x * gridDim.x;
	}
}

std::vector<Tensor> BatchNormalization::backPropagate(std::vector<Tensor>& input)
{
	std::vector<Tensor> dldstandr(input.size());
	std::transform(input.begin(), input.end(), dldstandr.begin(), 
		[this](Tensor& dldy) {return PerElementMult(dldy, this->gamma); });

	auto layerInputs = JoinTensors(this->batchInputs);
	auto standartaizedInputs = JoinTensors(this->standartized);
	auto gradientStandartaized = JoinTensors(dldstandr);

	uint dataSize = this->gamma.dataSize;
	uint batchSize = static_cast<uint>(input.size());
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	Tensor gradientVariance = GenerateFilledTensor(this->variance.width, this->variance.height, this->variance.channels, this->variance.four, 0);

	ComputeVarianceGradient<<<grids, threads>>>(layerInputs.data, gradientStandartaized.data,
												this->mean.data, this->variance.data, gradientVariance.data,
												dataSize, batchSize, EPS);

	Tensor gradientMean(this->mean.width, this->mean.height, this->mean.channels, this->mean.four);

	ComputeMeanGradient<<<grids, threads>>>(layerInputs.data, gradientStandartaized.data,
											this->mean.data, this->variance.data, gradientVariance.data, gradientMean.data,
											dataSize, batchSize, EPS);

	Tensor outputGradient(layerInputs.width, layerInputs.height, layerInputs.channels, layerInputs.four);

	ComputeOutputGradient<<<grids, threads>>>(layerInputs.data, gradientStandartaized.data,
											this->mean.data, gradientMean.data, 
											this->variance.data, gradientVariance.data, outputGradient.data,
											dataSize, batchSize, EPS);

	Tensor gradientGamma = GenerateFilledTensor(this->gamma.width, this->gamma.height, this->gamma.channels, this->gamma.four, 0);
	for (uint i = 0; i < batchSize; ++i)
	{
		gradientGamma += PerElementMult(input[i], this->standartized[i]);
	}

	Tensor gradientBeta = GenerateFilledTensor(this->beta.width, this->beta.height, this->beta.channels, this->beta.four, 0);
	for (uint i = 0; i < batchSize; ++i)
	{
		gradientBeta += input[i];
	}

	this->gamma -= gradientGamma * lambda;
	this->beta -= gradientBeta * lambda;

	return SplitTensors(outputGradient);
}

void BatchNormalization::print(std::ostream& out)
{

}

Json::Value BatchNormalization::Serialize()
{
	Json::Value json;

	return json;
}

void BatchNormalization::DeSerialize(Json::Value)
{
}


void BatchNormalization::Standardization(std::vector<Tensor>& input)
{
	double batchSize = static_cast<double>(input.size());
	uint width = input[0].width;
	uint height = input[0].height;
	uint channels = input[0].channels;
	uint four = input[0].four;

	this->mean = std::accumulate(input.begin(), input.end(), GenerateFilledTensor(width, height, channels, four, 0)) * (1. / batchSize);
	this->variance = std::accumulate(input.begin(), input.end(), GenerateFilledTensor(width, height, channels, four, 0),
		[this](Tensor& result, Tensor& value) {
			auto diff = value - this->mean;
			return result + PerElementMult(diff, diff);
		}) * (1. / batchSize);

	this->standartized.resize(input.size());
	std::transform(input.begin(), input.end(), standartized.begin(),
			[this](Tensor& value) {
				return PerElementDiv(value - this->mean, ApplySqrt(this->variance + EPS));
			});
}

Tensor BatchNormalization::Scale(Tensor& input)
{
	return PerElementMult(this->gamma, input) + this->beta;
}

Tensor BatchNormalization::ComputeGradient(Tensor& dldy)
{
	Tensor dldstandr = PerElementMult(dldy, this->gamma);

	return Tensor();
}