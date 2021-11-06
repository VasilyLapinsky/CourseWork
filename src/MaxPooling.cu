#include "MaxPooling.h"

const uint MAX_THREADS = 32;

MaxPooling::MaxPooling(uint pool, uint stride)
	: pool{pool}
	, stride{stride}
{
}

Tensor MaxPooling::compute(Tensor& input)
{
	this->inputs = input;
	return this->Forward(this->inputs);
}

std::vector<Tensor> MaxPooling::compute(std::vector<Tensor>& input)
{
	this->batchInputs = input;

	std::vector<Tensor> result;
	std::transform(this->batchInputs.begin(), this->batchInputs.end(), std::back_inserter(result),
		[this](Tensor& value) mutable { return this->Forward(value); });

	return result;
}

Tensor MaxPooling::backPropagate(Tensor& input)
{
	return this->ComputeGradient(this->inputs, input);
}

std::vector<Tensor> MaxPooling::backPropagate(std::vector<Tensor>& input)
{
	std::vector<Tensor> result(this->batchInputs.size());
	for (size_t i = 0; i < input.size(); ++i)
	{
		result[i] = this->ComputeGradient(this->batchInputs[i], input[i]);
	}

	return result;
}


void MaxPooling::print(std::ostream&)
{

}

Json::Value MaxPooling::Serialize()
{
	return Json::Value();
}

void MaxPooling::DeSerialize(Json::Value json)
{

}

__global__ void max_pool(double* data, double* result, const uint pool, const uint stride,
	const uint resultWidth, const uint resultHeight, const uint resultChannels, const uint resultFour,
	const uint resultRowSize, const uint resultChannelSize, const uint dataRowSize, const uint dataChannelSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < resultWidth && y < resultHeight)
	{
		uint dataX = x * stride;
		uint dataY = y * stride;
		uint numberOfAllChannels = resultChannels * resultFour;

		uint dataId = dataX + dataY * dataRowSize;
		uint resultId = x + y * resultRowSize;

		for (uint c = 0; c < numberOfAllChannels; ++c)
		{
			double maxValue = data[dataId];

			for (uint h = 0; h < pool; ++h)
			{
				for (uint w = 0; w < pool; ++w)
				{
					if (data[dataId] > maxValue)
					{
						maxValue = data[dataId];
					}

					++dataId;
				}
				dataId += dataRowSize - pool;
			}

			result[resultId] = maxValue;

			dataId += dataChannelSize - pool * dataRowSize;
			resultId += resultChannelSize;
		}
	}
}

Tensor MaxPooling::Forward(Tensor& input)
{
	uint resultWidth = (input.width - this->pool) / this->stride + 1;
	uint resultHeight = (input.height - this->pool) / this->stride + 1;

	Tensor result = GenerateFilledTensor(resultWidth, resultHeight, input.channels, input.four, 0);

	const uint inputRowSize = input.width;
	const uint inputChannelSize = input.height * inputRowSize;

	const uint resultRowSize = result.width;
	const uint resultChannelSize = result.height * resultRowSize;

	dim3 grids((result.width + MAX_THREADS - 1) / MAX_THREADS, (result.height + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS, MAX_THREADS);
	max_pool<<<grids, threads>>>(input.data, result.data, this->pool, this->stride,
		result.width, result.height, result.channels, result.four,
		resultRowSize, resultChannelSize, inputRowSize, inputChannelSize);
	HandleCudaStatus(cudaGetLastError());

	return result;
}

__global__ void compute_gradient(double* inputs, double* dy, double* result, const uint pool, const uint stride,
	const uint dyWidth, const uint dyHeight, const uint dyChannels, const uint dyFour,
	const uint dyRowSize, const uint dyChannelSize, const uint inputsRowSize, const uint inputsChannelSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dyWidth && y < dyHeight)
	{
		uint inputsX = x * stride;
		uint inputsY = y * stride;
		uint numberOfAllChannels = dyChannels * dyFour;

		uint inputsId = inputsX + inputsY * inputsRowSize;
		uint dyId = x + y * dyRowSize;

		for (uint c = 0; c < numberOfAllChannels; ++c)
		{
			double maxValue = inputs[inputsId];
			uint maxId = inputsId;

			for (uint h = 0; h < pool; ++h)
			{
				for (uint w = 0; w < pool; ++w)
				{
					if (inputs[inputsId] > maxValue)
					{
						maxValue = inputs[inputsId];
						maxId = inputsId;
					}

					++inputsId;
				}
				inputsId += inputsRowSize - pool;
			}

			result[maxId] = dy[dyId];

			inputsId += inputsChannelSize - pool * inputsRowSize;
			dyId += dyChannelSize;
		}
	}
}

Tensor MaxPooling::ComputeGradient(Tensor& layerInput, Tensor& gradient)
{
	Tensor result = GenerateFilledTensor(layerInput.width, layerInput.height, layerInput.channels, layerInput.four, 0);

	const uint inputRowSize = layerInput.width;
	const uint inputChannelSize = layerInput.height * inputRowSize;

	const uint gradientRowSize = gradient.width;
	const uint gradientChannelSize = gradient.height * gradientRowSize;

	dim3 grids((gradient.width + MAX_THREADS - 1) / MAX_THREADS, (gradient.height + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS, MAX_THREADS);
	compute_gradient<<<grids, threads>>>(layerInput.data, gradient.data, result.data, this->pool, this->stride,
		gradient.width, gradient.height, gradient.channels, gradient.four,
		gradientRowSize, gradientChannelSize, inputRowSize, inputChannelSize);

	return result;
}