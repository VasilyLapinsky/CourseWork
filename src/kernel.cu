#include "NeuralNet.h"
#include "MnistDatasetReader.h"
#include "StretchLayer.h"
#include "FullyConnected.h"
#include "ReLU.h"
#include "SoftMax.h"
#include "ConvLayer.h"
#include "MaxPooling.h"
#include "MathFunctions.h"
#include <iostream>
#include <json/json.h>

const std::string DATA_FOLDER = "C:/Course Work/data/";
const std::string CONFIG_FILE_PATH = "config.yaml";
const std::string WEIGHTS_FILE_PATH = "weights.bin";

Tensor ToTensor(std::vector<cv::Mat>& data)
{
	std::vector<Tensor> tensors;
	for (int i = 0; i < data.size(); ++i)
	{
		tensors.push_back(Tensor(data[i]));
	}
	auto tensor = JoinTensors(tensors);
	tensor.channels = tensor.four;
	tensor.four = 1;

	return tensor;
}

Tensor ToTensor(std::vector<std::vector<cv::Mat>>& data)
{
	std::vector<Tensor> tensors;
	for (int i = 0; i < data.size(); ++i)
	{
		tensors.push_back(ToTensor(data[i]));
	}
	return JoinTensors(tensors);
}

int main()
{
	NeuralNet net;

	std::unique_ptr<DatasetReaderInterface> datsetReader = std::make_unique<MnistDatasetReader>(DATA_FOLDER);
	net.train(datsetReader, 32, 1);

	/*
	double low = -500.0;
	double high = +500.0;

	ConvLayer conv(0.1, 5, 1, 5, 1, 0);
	std::cout << "Gpu weight: \n";
	PrintTensor(conv.weights);
	cpu::ConvLayer cpuConv(1, 5, 5, 1, 0.1, 0);
	std::cout << "Cpu weight: \n";
	PrintTensor(ToTensor(cpuConv.weights));

	cv::Mat left(24, 24, CV_64F);
	//cv::randu(left, cv::Scalar(low), cv::Scalar(high));
	left = 1.0;
	for (int i = 0; i < 10; ++i)
	{
		Tensor data = datsetReader->GetData().first;
		auto cpuResult = cpuConv.compute(std::vector<cv::Mat>({ TensorToCvMat(data) }));
		auto cpuResultOnGpu = ToTensor(cpuResult);
		auto gpuResult = conv.compute(data);

		auto diff = gpuResult - cpuResultOnGpu;
		std::cout << "Compute:\n" << Sum(diff) << '\n';

		cpuResult = cpuConv.backPropagate(cpuResult);
		cpuResultOnGpu = ToTensor(cpuResult);
		gpuResult = conv.backPropagate(gpuResult);

		diff = gpuResult - cpuResultOnGpu;
		std::cout << "BackPropagate:\n" << Sum(diff) << '\n';

		std::cout << "Weights:\n" << Sum(conv.weights - ToTensor(cpuConv.weights)) << '\n';
		std::cout << "Dw: \n";
		PrintTensor(conv.dwMemmory);
		PrintTensor(ToTensor(cpuConv.dwMemmory));
		std::cout << "Gpu bias:\n" << TensorToCvMat(conv.bias) << '\n';
		std::cout << "Cpu bias:\n";
		std::for_each(cpuConv.bias.begin(), cpuConv.bias.end(), [](double val) {std::cout << val << ' '; });
		std::cout << '\n';
	}
	*/
}