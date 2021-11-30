#include "NeuralNet/NeuralNet.h"
#include "DatasetReader/MnistDatasetReader.h"
#include "DatasetReader/RtsdDatasetReader.h"
#include "DatasetReader/UniversalDatasetReader.h"
#include "Visualization/NeuralNetVisualizer.h"
#include "NeuralNet/StretchLayer.h"
#include "NeuralNet/FullyConnected.h"
#include "NeuralNet/ReLU.h"
#include "NeuralNet/SoftMax.h"
#include "NeuralNet/ConvLayer.h"
#include "NeuralNet/MaxPooling.h"
#include "NeuralNet/MathFunctions.h"
#include <iostream>
#include <fstream>
#include <json/json.h>

const std::string DATA_FOLDER = "C:/Course Work/data";
const std::string MNIST_FOLDER = DATA_FOLDER + "/mnist";
const std::string RTSD_FOLDER = DATA_FOLDER + "/rtsd-r1";


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
	std::unique_ptr<DatasetReaderInterface> datsetReader = std::make_unique<UniversalDatasetReader>(ReadRTSD(RTSD_FOLDER));
	net.train(datsetReader, 64, 1);
	net.Save(CONFIG_FILE_PATH, WEIGHTS_FILE_PATH);

	// std::unique_ptr<DatasetReaderInterface> datsetReader = std::make_unique<UniversalDatasetReader>(ReadRTSD(RTSD_FOLDER));
	// NeuralNetVisualizer visualizer(CONFIG_FILE_PATH, WEIGHTS_FILE_PATH, std::move(datsetReader));
	// visualizer.RunVisualization();
	/*
	double low = -500.0;
	double high = +500.0;

	ConvLayer conv(0.1, 3, 1, 1, 1, 0);
	std::cout << "Gpu weight: \n";
	PrintTensor(conv.weights);
	cpu::ConvLayer cpuConv(1, 3, 1, 1, 0.1, 0);
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
	
	ConvLayer conv(0.1, 5, 1, 3, 1, 0);
	Tensor data = GenerateUniformDistributionTensor(48, 48, 1, 1);
	
	std::cout << "Weights:\n";
	PrintTensor(conv.weights);
	std::cout << "data:\n";
	PrintTensor(data);
	std::cout << "forward:\n";
	auto forward = conv.compute(data);
	PrintTensor(forward);
	std::cout << "backward:\n";
	auto backward = conv.backPropagate(forward);
	PrintTensor(backward);
	
}