#include "Tensor.h"
#include "curand.h"
#include <iostream>

const uint MAX_THREADS = 32;

Tensor::Tensor()
	: width{0}
	, height{0}
	, channels{0}
	, four{0}
	, dataSize{0}
	, data{nullptr}
{
}

Tensor::Tensor(uint width, uint height, uint channels, uint four)
	: width{width}
	, height{height}
	, channels{channels}
	, four{four}
	, dataSize{width * height * channels * four}
{
	HandleCudaStatus(cudaMalloc((void**)&this->data, dataSize * sizeof(double)));
}

Tensor::Tensor(const cv::Mat& value)
	: width{ static_cast<uint>(value.cols)}
	, height{ static_cast<uint>(value.rows)}
	, channels{ static_cast<uint>(value.channels()) }
	, four{1}
	, dataSize{ static_cast<uint>(value.channels() * value.cols * value.rows)}
{
	HandleCudaStatus(cudaMalloc((void**)&(this->data), dataSize * sizeof(double)));
	HandleCudaStatus(cudaMemcpy((void*)(this->data), (void*)value.data, dataSize * sizeof(double), cudaMemcpyHostToDevice));
}

Tensor::Tensor(Json::Value json)
{
	this->width = json["Tensor"]["width"].asInt();
	this->height = json["Tensor"]["height"].asInt();
	this->channels = json["Tensor"]["channels"].asInt();
	this->four = json["Tensor"]["four"].asInt();
	this->dataSize = json["Tensor"]["dataSize"].asInt();

	double* dataOnCpu = new double[this->dataSize];

	Json::Value array = json["Tensor"]["data"];
	for (uint i = 0; i < this->dataSize; ++i)
	{
		dataOnCpu[i] = array[i].asDouble();
	}

	HandleCudaStatus(cudaMalloc((void**)&(this->data), dataSize * sizeof(double)));
	HandleCudaStatus(cudaMemcpy((void*)(this->data), (void*)dataOnCpu, this->dataSize * sizeof(double), cudaMemcpyHostToDevice));
}

Tensor::Tensor(const Tensor& value)
	: width{value.width }
	, height{value.height}
	, channels{value.channels}
	, four {value.four}
	, dataSize{value.dataSize}
{
	HandleCudaStatus(cudaMalloc((void**)&(this->data), dataSize * sizeof(double)));
	HandleCudaStatus(cudaMemcpy((void*)(this->data), (void*)value.data, dataSize * sizeof(double), cudaMemcpyDeviceToDevice));
}

Tensor::Tensor(Tensor&& value)
	: width{ value.width }
	, height{ value.height }
	, channels{ value.channels }
	, four { value.four }
	, dataSize{ value.dataSize }
{
	this->data = value.data;
	value.data = nullptr;
}

Tensor::~Tensor()
{
	if (this->data != nullptr)
	{
		HandleCudaStatus(cudaGetLastError());
		cudaFree(this->data);
		HandleCudaStatus(cudaGetLastError());
	}
}

__global__ void add(double* data, double value, double* result, uint size)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		result[tid] = data[tid] + value;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void mult(double* data, double value, double* result, uint size)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		result[tid] = data[tid] * value;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void add(double* left, double* right, double* result, uint size) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		result[tid] = left[tid] + right[tid];
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void sub(double* left, double* right, double* result, uint size) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		result[tid] = left[tid] - right[tid];
		tid += blockDim.x * gridDim.x;
	}
}


void Tensor::operator+=(double value)
{
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	add<<<grids, threads>>>(this->data, value, this->data, dataSize);

	HandleCudaStatus(cudaGetLastError());
}

Tensor Tensor::operator+(double value) const
{
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	Tensor result(this->width, this->height, this->channels, this->four);
	add<<<grids, threads>>>(this->data, value, result.data, dataSize);

	HandleCudaStatus(cudaGetLastError());

	return result;
}

void Tensor::operator*=(double value)
{
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	mult<<<grids, threads>>>(this->data, value, this->data, dataSize);

	HandleCudaStatus(cudaGetLastError());
}

Tensor Tensor::operator*(double value) const
{
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	Tensor result(this->width, this->height, this->channels, this->four);
	mult<<<grids, threads>>>(this->data, value, result.data, dataSize);

	HandleCudaStatus(cudaGetLastError());

	return result;
}

void Tensor::operator+=(Tensor& value)
{
	if (this->four != value.four
		||this->channels != value.channels 
		|| this->width != value.width 
		|| this->height != value.height)
	{
		throw std::exception("Invalid size!");
	}
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	add<<<grids, threads>>>(this->data, value.data, this->data, dataSize);

	HandleCudaStatus(cudaGetLastError());
}

Tensor Tensor::operator+(Tensor& value) const
{
	if (this->four != value.four
		|| this->channels != value.channels
		|| this->width != value.width
		|| this->height != value.height)
	{
		throw std::exception("Invalid size!");
	}
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	Tensor result(this->width, this->height, this->channels, this->four);
	add<<<grids, threads>>>(this->data, value.data, result.data, dataSize);

	HandleCudaStatus(cudaGetLastError());

	return result;
}

void Tensor::operator-=(Tensor& value)
{
	if (this->four != value.four
		|| this->channels != value.channels
		|| this->width != value.width
		|| this->height != value.height)
	{
		throw std::exception("Invalid size!");
		return;
	}
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	sub<<<grids, threads>>>(this->data, value.data, this->data, dataSize);

	HandleCudaStatus(cudaGetLastError());
}

Tensor Tensor::operator-(Tensor& value) const
{
	if (this->four != value.four
		|| this->channels != value.channels
		|| this->width != value.width
		|| this->height != value.height)
	{
		throw std::exception("Invalid size!");
		return Tensor();
	}
	dim3 grids((dataSize + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);

	Tensor result(this->width, this->height, this->channels, this->four);
	sub<<<grids, threads>>>(this->data, value.data, result.data, dataSize);

	HandleCudaStatus(cudaGetLastError());

	return result;
}

Tensor Tensor::operator=(const Tensor& value)
{
	if (this->data == value.data)
	{
		return *this;
	}

	if (this->dataSize != value.dataSize)
	{
		cudaFree(this->data);
		HandleCudaStatus(cudaMalloc((void**)&(this->data), value.dataSize * sizeof(double)));
	}

	this->width = value.width;
	this->height = value.height;
	this->channels = value.channels;
	this->four = value.four;
	this->dataSize = value.dataSize;

	HandleCudaStatus(cudaMemcpy((void*)(this->data), (void*)value.data, this->dataSize * sizeof(double), cudaMemcpyDeviceToDevice));

	return *this;
}

cv::Mat TensorToCvMat(Tensor& value)
{
	if (value.four != 1)
	{
		throw std::exception(std::string("can not transform Tensor to cvMat with multi four dimmensions: "
			+ std::to_string(value.four)).c_str());
	}

	cv::Mat result;
	if (value.channels == 1)
	{
		result = cv::Mat(value.height, value.width, CV_64F);
	}
	else if (value.channels == 2)
	{
		result = cv::Mat(value.height, value.width, CV_64FC2);
	}
	else if (value.channels == 3)
	{
		result = cv::Mat(value.height, value.width, CV_64FC3);
	}
	else
	{
		throw std::exception(std::string("can not transform Tensor to cvMat as number of channels: " 
							+ std::to_string(value.channels)).c_str());
	}

	HandleCudaStatus(cudaMemcpy((void*)result.data, (void*)value.data, value.dataSize * sizeof(double), cudaMemcpyDeviceToHost));

	return result;
}

Json::Value TensorToJson(Tensor& value)
{
	Json::Value data;
	data["Tensor"]["width"] = value.width;
	data["Tensor"]["height"] = value.height;
	data["Tensor"]["channels"] = value.channels;
	data["Tensor"]["four"] = value.four;
	data["Tensor"]["dataSize"] = value.dataSize;

	double* dataOnCpu = new double[value.dataSize];
	HandleCudaStatus(cudaMemcpy((void*)dataOnCpu, (void*)value.data, value.dataSize * sizeof(double), cudaMemcpyDeviceToHost));

	Json::Value array(Json::arrayValue);
	for (uint i = 0; i < value.dataSize; ++i)
	{
		array.append(dataOnCpu[i]);
	}

	data["Tensor"]["data"] = array;
	
	return data;
}

curandGenerator_t CreateGenerator()
{
	curandGenerator_t generator;

	/* Create pseudo-random number generator */
	HandleCudaRandStatus(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));

	/* Set seed */
	HandleCudaRandStatus(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));

	return generator;
}

Tensor GenerateUniformDistributionTensor(uint width, uint height, uint channels, uint four)
{
	Tensor result(width, height, channels, four);

	auto randomNumberGenerator = CreateGenerator();
	HandleCudaRandStatus(curandGenerateUniformDouble(randomNumberGenerator, result.data, result.dataSize));

	return result;
}

Tensor GenerateNormalDistributionTensor(uint width, uint height, uint channels, uint four, double mean, double stddev)
{
	Tensor result(width, height, channels, four);

	auto randomNumberGenerator = CreateGenerator();
	if (result.dataSize % 2 == 0)
	{
		HandleCudaRandStatus(curandGenerateNormalDouble(randomNumberGenerator, result.data, result.dataSize, mean, stddev));
	}
	else
	{
		double* temp;
		HandleCudaStatus(cudaMalloc((void**)&temp, (result.dataSize+1) * sizeof(double)));
		HandleCudaRandStatus(curandGenerateNormalDouble(randomNumberGenerator, temp, result.dataSize+1, mean, stddev));
		HandleCudaStatus(cudaMemcpy((void*)(result.data), (void*)temp, result.dataSize * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaFree(temp);
	}

	return result;
}


__global__ void fill(double* matrix, uint size, double val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (uint i = idx; i < size; i += gridDim.x * blockDim.x) {
		matrix[i] = val;
	}
}

Tensor GenerateFilledTensor(uint width, uint height, uint channels, uint four, double value)
{
	Tensor result(width, height, channels, four);

	const uint BLOCK_SIZE = 256;
	const uint NUM_BLOCKS = (result.dataSize + BLOCK_SIZE - 1) / (result.dataSize);
	fill<<<NUM_BLOCKS, BLOCK_SIZE>>>(result.data, result.dataSize, value);

	return result;
}

Tensor JoinTensors(std::vector<Tensor>& value)
{
	if (value.size() == 0)
	{
		return Tensor();
	}

	if (value[0].four != 1)
	{
		throw std::invalid_argument("Tensors must have only 3 dimensional!");
	}

	Tensor result(value[0].width, value[0].height, value[0].channels, value.size());
	for (uint i = 0; i < result.four; ++i)
	{
		double* resultPtr = i * result.width * result.height * result.channels + result.data;
		HandleCudaStatus(cudaMemcpy((void*)resultPtr, (void*)value[i].data, value[i].dataSize * sizeof(double), cudaMemcpyHostToHost));
	}

	return result;
}

std::vector<Tensor> SplitTensors(Tensor& value)
{
	std::vector<Tensor> result;
	if (value.four == 0 || value.dataSize == 0)
	{
		return result;
	}

	uint dataSize = value.dataSize / value.four;
	for (uint i = 0; i < value.four; ++i)
	{
		Tensor output(value.width, value.height, value.channels, 1);
		HandleCudaStatus(cudaMemcpy((void*)output.data, (void*)(value.data + i * dataSize), dataSize*sizeof(double), cudaMemcpyHostToHost));

		result.push_back(output);
	}

	return result;
}

void PrintTensor(Tensor& value)
{
	double* cpudata = new double[value.dataSize];
	cudaMemcpy(cpudata, value.data, value.dataSize * sizeof(double), cudaMemcpyDeviceToHost);
	uint id = 0;
	for (uint f = 0; f < value.four; ++f)
	{
		std::cout << "|\n";
		for (uint c = 0; c < value.channels; ++c)
		{
			std::cout << "[\n";
			for (uint h = 0; h < value.height; ++h)
			{
				for (uint w = 0; w < value.width; ++w)
				{
					std::cout << cpudata[id] << ' ';
					++id;
				}
				std::cout << ";\n";
			}
			std::cout << "]\n";
		}
		std::cout << "|\n";
	}

	delete[] cpudata;
}
