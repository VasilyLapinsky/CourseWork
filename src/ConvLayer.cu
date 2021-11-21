#include "ConvLayer.h"
#include "MathFunctions.h"
#include "device_functions.h"
#include <iostream>

const char* ConvLayerConfigNodeName = "ConvLayer";

const uint MAX_THREADS = 32;

ConvLayer::ConvLayer(double lambda, uint kernelSize, uint inputChannels,
	uint numFilters, uint stride, uint padding)
	: lambda{ lambda }
	, stride{ stride }
	, padding{ padding }
	, kernelSize{ kernelSize }
	, weights{ GenerateNormalDistributionTensor(kernelSize, kernelSize, inputChannels, numFilters,
												0, 1.0 / sqrt(numFilters * kernelSize * kernelSize)) }
	, bias{ GenerateFilledTensor(numFilters, 1, 1, 1, 0) }
{

}

Tensor ConvLayer::compute(Tensor& input)
{
	return this->ConvolveTensor(input);
}

std::vector<Tensor> ConvLayer::compute(std::vector<Tensor>& input)
{
	return SplitTensors(this->ConvolveTensor(JoinTensors(input)));
}

Tensor Padding(Tensor& value, uint padding);

__global__ void convolve(double* filters, double* data, double* result, const uint stride,
	const uint filtersWidth, const uint filtersHeight, const uint filtersChannels, const uint filtersFour, const uint filtersFourSize,
	const uint dataRowSize, const uint dataCannelsSize, const uint dataFourSize,
	const uint resultWidth, const uint resultHeight, const uint resultChannels, const uint resultFour,
	const uint resultChannelsSize, const uint resultFourSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < resultWidth && y < resultHeight)
	{
		uint dataX = x * stride;
		uint dataY = y * stride;

		for (uint r = 0; r < resultFour; ++r)
		{
			for (uint k = 0; k < resultChannels; ++k)
			{
				uint filterId = k * filtersFourSize;
				uint inputId = dataX + dataY * dataRowSize + r * dataFourSize;
				double value = 0;

				for (uint c = 0; c < filtersChannels; ++c)
				{
					for (uint h = 0; h < filtersHeight; ++h)
					{
						for (uint w = 0; w < filtersWidth; ++w)
						{
							value += filters[filterId] * data[inputId];
							++filterId;
							++inputId;
						}
						inputId += dataRowSize - filtersWidth;
					}
					inputId += dataCannelsSize - dataRowSize * filtersHeight;
				}

				uint resultId = x + y * resultWidth + k * resultChannelsSize + r * resultFourSize;
				result[resultId] = value;
			}

		}
	}
}

__global__ void addBias(double* result, double* bias, const uint resultChannelSize, const uint resultFour, const uint resultFourSize)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < resultFourSize) {
		double value = bias[tid / resultChannelSize];
		for (uint i = 0; i < resultFour; ++i)
		{
			result[tid + i * resultFourSize] += value;
		}
		tid += blockDim.x * gridDim.x;
	}
}

Tensor ConvLayer::ConvolveTensor(Tensor& value)
{
	if (value.channels != this->weights.channels)
	{
		throw std::invalid_argument("Number of inputs channels must be the same as number of filters channels");
	}

	this->inputs = Padding(value, this->padding);

	const uint weightsFourSize = this->weights.width * this->weights.height * this->weights.channels;

	uint resultWidth = (this->inputs.width - this->kernelSize) / this->stride + 1;
	uint resultHeight = (this->inputs.height - this->kernelSize) / this->stride + 1;
	uint resultChannels = this->weights.four;
	uint resultFour = value.four;

	Tensor result(resultWidth, resultHeight, resultChannels, resultFour);

	uint dataRowSize = this->inputs.width;
	uint dataCannelsSize = this->inputs.width * this->inputs.height;
	uint dataFourSize = dataCannelsSize * this->inputs.channels;

	uint resultChannelsSize = result.width * result.height;
	uint resultFourSize = resultChannelsSize * result.channels;

	dim3 grids((result.width + MAX_THREADS - 1) / MAX_THREADS, (result.height + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS, MAX_THREADS);
	convolve << <grids, threads >> > (this->weights.data, this->inputs.data, result.data, this->stride,
		this->weights.width, this->weights.height, this->weights.channels, this->weights.four, weightsFourSize,
		dataRowSize, dataCannelsSize, dataFourSize,
		result.width, result.height, result.channels, result.four,
		resultChannelsSize, resultFourSize);
	HandleCudaStatus(cudaGetLastError());

	grids = dim3((resultFourSize + MAX_THREADS - 1) / MAX_THREADS);
	threads = dim3(MAX_THREADS);
	addBias << <grids, threads >> > (result.data, this->bias.data, resultChannelsSize, result.four, resultFourSize);
	HandleCudaStatus(cudaGetLastError());

	return result;
}

Tensor BackPropagatePadding(Tensor& value, uint padding);

__device__ void calculateDW(double* dw, double* inputs, double coeff, const uint kernelSize, const uint inputsRowSize)
{
	uint dwId = 0;
	uint inputsId = 0;
	for (uint h = 0; h < kernelSize; ++h)
	{
		for (uint w = 0; w < kernelSize; ++w)
		{

			dw[dwId] += coeff * inputs[inputsId];
			++dwId;
			++inputsId;
		}
		inputsId += inputsRowSize - kernelSize;
	}
}

__global__ void back_propagate_dw(double* inputs, double* dy, double* dw,
	const uint stride, const uint kernelSize,
	const uint dyWidth, const uint dyHeight, const uint filtersChannels, const uint filtersFour,
	const uint inputsRowSize, const uint inputsChannelSize, const uint dyChannelSize,
	const uint filtersChannelSize, const uint filtersFourSize)
{
	int filter = blockIdx.y * blockDim.y + threadIdx.y;
	int channel = blockIdx.x * blockDim.x + threadIdx.x;

	if (filter < filtersFour && channel < filtersChannels)
	{
		double* dwPointer = dw + channel * filtersChannelSize + filter * filtersFourSize;
		uint dyId = filter * dyChannelSize;

		for (uint h = 0; h < dyHeight; ++h)
		{
			for (uint w = 0; w < dyWidth; ++w)
			{
				uint inputsId = w * stride + h * stride * inputsRowSize + channel * inputsChannelSize;
				double coeff = dy[dyId];
				calculateDW(dwPointer, inputs + inputsId, coeff, kernelSize, inputsRowSize);
				++dyId;
			}
		}
	}
}

__global__ void back_propagate_dx(double* filters, double* dy, double* dx,
	const uint stride, const int kernelSize,
	const uint filtersChannels, const uint filtersFour,
	const uint dxWidth, const uint dxHeight, const uint dxChannels,
	const uint dxRowSize, const uint dxChannelSize, const uint dyRowSize, const uint dyChannelSize, const uint dyDataSize,
	const uint filtersRowSize, const uint filtersChannelSize, const uint filtersFourSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dxWidth && y < dxHeight)
	{
		uint dxId = x + y * dxRowSize;
		uint dyId = 0;
		uint width = 0;
		uint height = 0;
		uint filterId = 0;

		if (x < kernelSize || y < kernelSize)
		{
			dyId = max(0, (x - kernelSize + 1)) + max(0, (y - kernelSize + 1)) * dyRowSize;
			width = umin(kernelSize, x+1) / stride;
			height = umin(kernelSize, y+1) / stride;
			filterId = 0;
		}
		else if ((dxWidth - x) < kernelSize || (dxHeight - y) < kernelSize)
		{
			if ((x / stride) + kernelSize > dxWidth || (y / stride) + kernelSize > dxWidth)
			{
				return;
			}

			dyId = (x - kernelSize + 1) + (y - kernelSize + 1) * dyRowSize;
			width = umin(kernelSize, dxWidth - x + 1) / stride;
			height = umin(kernelSize, dxHeight - y + 1) / stride;
			filterId = (kernelSize - dxWidth + x) + (kernelSize - dxHeight + y) * filtersRowSize;
			
		}
		else
		{
			dyId = (x - kernelSize + 1) + (y - kernelSize + 1) * dyRowSize;
			width = kernelSize / stride;
			height = kernelSize / stride;
			filterId = 0;
		}

		for (uint f = 0; f < filtersFour; ++f)
		{
			for (uint h = 0; h < height; ++h)
			{
				for (uint w = 0; w < width; ++w)
				{
					if (dyId >= dyDataSize)
					{
						return;
					}
					double coeff = dy[dyId];
					for (uint c = 0; c < filtersChannels; ++c)
					{
						dx[dxId + c * dxChannelSize] += coeff * filters[filterId + c * filtersChannelSize];
					}
					++dyId;
					filterId += stride;
				}
				dyId += dyRowSize - width;
				filterId += stride * filtersRowSize - width;
			}
			filterId += filtersFourSize - stride * height * filtersRowSize;
			dyId += dyChannelSize - height * dyRowSize;
		}
	}
}


Tensor ConvLayer::backPropagate(Tensor& dy)
{
	Tensor dw = GenerateFilledTensor(this->weights.width, this->weights.height, this->weights.channels, this->weights.four, 0);
	Tensor dx = GenerateFilledTensor(this->inputs.width, this->inputs.height, this->inputs.channels, this->inputs.four, 0);

	const uint inputsRowSize = this->inputs.width;
	const uint inputsChannelSize = this->inputs.height * inputsRowSize;

	const uint dyChannelSize = dy.width * dy.height;
	const uint filtersChannelSize = this->weights.width * this->weights.height;
	const uint filtersFourSize = this->weights.channels * filtersChannelSize;

	dim3 grids((dw.channels + MAX_THREADS - 1) / MAX_THREADS, (dw.four + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(std::min(dw.channels, MAX_THREADS), std::min(dw.four, MAX_THREADS));

	back_propagate_dw<<<grids, threads>>>(this->inputs.data, dy.data, dw.data,
		this->stride, this->kernelSize, dy.width, dy.height, this->weights.channels, this->weights.four,
		inputsRowSize, inputsChannelSize, dyChannelSize, filtersChannelSize, filtersFourSize);

	HandleCudaStatus(cudaGetLastError());
	const uint dxChannelSize = dx.width * dx.height;

	grids = dim3((dx.width + MAX_THREADS - 1) / MAX_THREADS, (dx.height + MAX_THREADS - 1) / MAX_THREADS);
	threads = dim3(std::min(dx.width, MAX_THREADS), std::min(dx.height, MAX_THREADS));
	back_propagate_dx<<<grids, threads>>>(this->weights.data, dy.data, dx.data,
		this->stride, this->kernelSize, this->weights.channels, this->weights.four,
		dx.width, dx.height, dx.channels, dx.width, dxChannelSize, dy.width, dyChannelSize, dy.dataSize,
		this->weights.width, filtersChannelSize, filtersFourSize);

	HandleCudaStatus(cudaGetLastError());
	Tensor db = SumForChannels(dy);

	this->weights -= dw * this->lambda;
	this->dwMemmory = dw;
	this->bias -= db * this->lambda;

	return BackPropagatePadding(dx, this->padding);
}

std::vector<Tensor> ConvLayer::backPropagate(std::vector<Tensor>& dy)
{
	const uint bathcSize = dy.size();
	Tensor batchDw = GenerateFilledTensor(this->weights.width, this->weights.height, this->weights.channels, this->weights.four, 0);
	Tensor batchDb = GenerateFilledTensor(this->bias.width, this->bias.height, this->bias.channels, this->bias.four, 0);
	std::vector<Tensor> batchDx;

	const uint fullKernelSize = this->kernelSize * this->kernelSize;
	const uint inputsRowSize = this->inputs.width;
	const uint inputsChannelSize = this->inputs.height * inputsRowSize;
	const uint inputFourSize = this->inputs.channels * inputsChannelSize;

	const uint dyChannelSize = dy[0].width * dy[0].height;
	const uint filtersChannelSize = this->weights.width * this->weights.height;
	const uint filtersFourSize = this->weights.channels * filtersChannelSize;
	const uint dxChannelSize = this->inputs.width * this->inputs.height;

	dim3 grids;
	dim3 threads;

	Tensor dw;
	Tensor dx;
	Tensor db;

	for (uint i = 0; i < bathcSize; ++i)
	{
		try
		{
			dw = GenerateFilledTensor(this->weights.width, this->weights.height, this->weights.channels, this->weights.four, 0);
			dx = GenerateFilledTensor(this->inputs.width, this->inputs.height, this->inputs.channels, 1, 0);

			double* inputsDataPointer = this->inputs.data + i * inputFourSize;
			double* dyDataPointer = dy[i].data;

			grids = ((this->weights.channels + MAX_THREADS - 1) / MAX_THREADS, (this->weights.four + MAX_THREADS - 1) / MAX_THREADS);
			threads = (std::min(this->weights.channels, MAX_THREADS), std::min(this->weights.four, MAX_THREADS));
			back_propagate_dw<<<grids, threads>>>(inputsDataPointer, dyDataPointer, dw.data,
				this->stride, fullKernelSize, dy[i].width, dy[i].height, this->weights.channels, this->weights.four,
				inputsRowSize, inputsChannelSize, dyChannelSize, filtersChannelSize, filtersFourSize);

			HandleCudaStatus(cudaGetLastError());

			grids = dim3((dx.width + MAX_THREADS - 1) / MAX_THREADS, (dx.height + MAX_THREADS - 1) / MAX_THREADS);
			threads = dim3(std::min(dx.width, MAX_THREADS), std::min(dx.height, MAX_THREADS));
			back_propagate_dx<<<grids, threads>>>(this->weights.data, dyDataPointer, dx.data,
				this->stride, this->kernelSize, this->weights.channels, this->weights.four,
				dx.width, dx.height, dx.channels, dx.width, dxChannelSize, dy[i].width, dyChannelSize, dy[i].dataSize,
				this->weights.width, filtersChannelSize, filtersFourSize);

			HandleCudaStatus(cudaGetLastError());

			batchDw += dw;
			dx = BackPropagatePadding(dx, this->padding);
			batchDx.push_back(dx);
			db = SumForChannels(dy[i]);
			batchDb += db;
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	// std::cout << " sum dw before:" << Sum(batchDw) << " \n";

	batchDw *= 1.0 / static_cast<double>(bathcSize);
	batchDb *= 1.0 / static_cast<double>(bathcSize);

	/*
	std::cout << " sum dw after:" << Sum(batchDw) << " \n";
	std::cout << " sum dw expected:" << Sum(batchDw) * static_cast<double>(bathcSize) << " \n";
	std::cout << " sum dx: " << Sum(JoinTensors(batchDx)) << " \n";
	//PrintTensor(JoinTensors(batchDx));
	std::cout << " sum weights: " << Sum(this->weights) << " \n";
	std::cout << " max dx: " << Sum(this->weights) * Sum(JoinTensors(dy)) << " \n";

	// std::cout << "sum dy:" << Sum(JoinTensors(dy)) <<" \n";
	// std::cout << "sum inputs:" << Sum(this->inputs) << " \n";
	// std::cout << "batchDb: \n";
	// PrintTensor(batchDw);
	*/

	this->weights -= batchDw * this->lambda;
	this->bias -= batchDb * this->lambda;

	return batchDx;

}

void ConvLayer::print(std::ostream&)
{

}

void ConvLayer::Serialize(Json::Value& config, std::ofstream& weigths)
{
	config["name"] = ConvLayerConfigNodeName;
	config[ConvLayerConfigNodeName]["lambda"] = this->lambda;
	config[ConvLayerConfigNodeName]["stride"] = this->stride;
	config[ConvLayerConfigNodeName]["padding"] = this->padding;
	config[ConvLayerConfigNodeName]["kernelSize"] = this->kernelSize;
	this->weights.Serrialize(config[ConvLayerConfigNodeName]["weights"], weigths);
	this->bias.Serrialize(config[ConvLayerConfigNodeName]["bias"], weigths);
}

void ConvLayer::DeSerialize(Json::Value& config, std::ifstream& weigths)
{
	this->lambda = config[ConvLayerConfigNodeName]["lambda"].asDouble();
	this->stride = config[ConvLayerConfigNodeName]["stride"].asUInt();
	this->padding = config[ConvLayerConfigNodeName]["padding"].asUInt();
	this->kernelSize = config[ConvLayerConfigNodeName]["kernelSize"].asUInt();
	this->weights = Tensor(config[ConvLayerConfigNodeName]["weights"], weigths);
	this->bias = Tensor(config[ConvLayerConfigNodeName]["bias"], weigths);
}

__global__ void pad(double* value, double* result, const uint padding,
	const uint valueRowSize, const uint valueChannelSize,
	const uint resultRowSize, const uint resultChannelSize,
	const uint resultWidth, const uint resultHeight, const uint resultChannels, const uint resultFour)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < resultWidth && y < resultHeight)
	{
		const uint allNumberOfChannels = resultChannels * resultFour;
		uint resultId = x + y * resultWidth;
		if (x >= padding && y >= padding && x < resultWidth - padding && y < resultHeight - padding)
		{
			uint valueId = (x - padding) + (y - padding) * valueRowSize;
			for (uint i = 0; i < allNumberOfChannels; ++i)
			{
				result[resultId] = value[valueId];

				resultId += resultChannelSize;
				valueId += valueChannelSize;
			}

		}
		else
		{
			for (uint i = 0; i < allNumberOfChannels; ++i)
			{
				result[resultId] = 0;

				resultId += resultChannelSize;
			}
		}
	}
}

Tensor Padding(Tensor& value, uint padding)
{
	if (padding == 0)
	{
		return value;
	}

	Tensor result(value.width + padding * 2, value.height + padding * 2, value.channels, value.four);

	uint valueRowsSize = value.width;
	uint valueChannelsSize = value.width * value.height;

	uint resultRowsSize = result.width;
	uint resultChannelsSize = result.width * result.height;

	dim3 grids((result.width + MAX_THREADS - 1) / MAX_THREADS, (result.height + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS, MAX_THREADS);

	pad<<<grids, threads>>>(value.data, result.data, padding, valueRowsSize, valueChannelsSize,
		resultRowsSize, resultChannelsSize, result.width, result.height, result.channels, result.four);

	HandleCudaStatus(cudaGetLastError());

	return result;
}

__global__ void backpropagate_pad(double* value, double* result, const uint padding,
	const uint valueRowSize, const uint valueChannelSize,
	const uint resultRowSize, const uint resultChannelSize,
	const uint resultWidth, const uint resultHeight, const uint resultChannels, const uint resultFour)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < resultWidth && y < resultHeight)
	{
		uint valueId = (x + padding) + (y + padding) * valueRowSize;
		uint resultId = x + y * resultRowSize;

		const uint allNumberOfChannels = resultChannels * resultFour;
		for (uint i = 0; i < allNumberOfChannels; ++i)
		{
			result[resultId] = value[valueId];
			resultId += resultChannelSize;
			valueId += valueChannelSize;
		}

	}
}

Tensor BackPropagatePadding(Tensor& value, uint padding)
{
	if (padding == 0)
	{
		return value;
	}

	Tensor result(value.width - padding * 2, value.height - padding * 2, value.channels, value.four);

	uint valueRowsSize = value.width;
	uint valueChannelsSize = value.width * value.height;

	uint resultRowsSize = result.width;
	uint resultChannelsSize = result.width * result.height;

	dim3 grids((result.width + MAX_THREADS - 1) / MAX_THREADS, (result.height + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS, MAX_THREADS);

	backpropagate_pad<<<grids, threads>>>(value.data, result.data, padding, valueRowsSize, valueChannelsSize,
		resultRowsSize, resultChannelsSize, result.width, result.height, result.channels, result.four);

	HandleCudaStatus(cudaGetLastError());

	return result;
}