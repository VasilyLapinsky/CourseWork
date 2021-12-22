#include "NeuralNet/NeuralNet.h"
#include "NeuralNet/MathFunctions.h"
#include "NeuralNet/ConvLayer.h"
#include "NeuralNet/StretchLayer.h"
#include "NeuralNet/FullyConnected.h"
#include "NeuralNet/ReLU.h"
#include "NeuralNet/SoftMax.h"
#include "NeuralNet/BatchNormalization.h"
#include "NeuralNet/MaxPooling.h"
#include "NeuralNet/LayersBuilder.h"
#include <iostream>
#include <fstream>

//---------------------------------------------------------

const char* NetConfurationNode = "NetConfiguration";
const size_t NUMBER_OF_CLASSES = 67;

//---------------------------------------------------------

NeuralNet::NeuralNet()
{
	double lr = 0.5;
	/*
	this->layers.push_back(std::make_shared<BatchNormalization>(lr, 28, 28));
	this->layers.push_back(std::make_shared<ConvLayer>(lr, 5, 1, 1, 1, 0));
	this->layers.push_back(std::make_shared<StretchLayer>(24, 24));
	this->layers.push_back(std::make_shared<FullyConnected>(576, 32, lr));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<FullyConnected>(32, 10, lr));
	this->layers.push_back(std::make_shared<SoftMax>());
	*/
	this->layers.push_back(std::make_shared<BatchNormalization>(lr, 48, 48));
	this->layers.push_back(std::make_shared<ConvLayer>(lr, 5, 1, 6, 1, 0));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<MaxPooling>(2, 2));
	// this->layers.push_back(std::make_shared<BatchNormalization>(lr, 14, 14, 6));
	this->layers.push_back(std::make_shared<ConvLayer>(lr, 5, 6, 16, 1, 0));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<MaxPooling>(2, 2));
	this->layers.push_back(std::make_shared<ConvLayer>(lr, 5, 16, 120, 1, 0));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<MaxPooling>(2, 2));
	this->layers.push_back(std::make_shared<ConvLayer>(lr, 2, 120, 480, 1, 0));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<StretchLayer>(1, 1, 480));
	this->layers.push_back(std::make_shared<FullyConnected>(480, 240, lr));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(std::make_shared<FullyConnected>(240, NUMBER_OF_CLASSES, lr));
	this->layers.push_back(std::make_shared<SoftMax>());

	/*
	auto fc1 = std::make_shared<FullyConnected>(784, 32, lr);
	auto cpufc1 = std::make_shared<cpu::FullyConnected>(784, 32, lr);
	cpufc1->weights = TensorToCvMat(fc1->weights);

	auto fc2 = std::make_shared<FullyConnected>(32, 10, lr);
	auto cpufc2 = std::make_shared<cpu::FullyConnected>(32, 10, lr);
	cpufc2->weights = TensorToCvMat(fc2->weights);

	this->layers.push_back(std::make_shared<BatchNormalization>(lr, 28, 28));
	this->layers.push_back(std::make_shared<StretchLayer>(28, 28));
	this->layers.push_back(fc1);
	this->layers.push_back(std::make_shared<BatchNormalization>(lr, 32, 1));
	this->layers.push_back(std::make_shared<ReLU>());
	this->layers.push_back(fc2);
	this->layers.push_back(std::make_shared<SoftMax>());
	*/
	/*this->cpuLayers.push_back(std::make_shared<cpu::StretchLayer>());
	this->cpuLayers.push_back(cpufc1);
	this->cpuLayers.push_back(std::make_shared<cpu::ReLU>());
	this->cpuLayers.push_back(cpufc2);
	this->cpuLayers.push_back(std::make_shared<cpu::SoftMax>());*/
}

NeuralNet::NeuralNet(std::string configfilePath, std::string weightsPath)
{
	this->Load(configfilePath, weightsPath);
}

void NeuralNet::addLayer(std::shared_ptr<LayerInterface> layer)
{
	layer->print(std::cout);
	this->layers.push_back(layer);
}

void NeuralNet::Save(std::string configfilePath, std::string weightsPath)
{
	Json::Value config;
	std::ofstream weights(weightsPath, std::ios::binary);

	for (size_t i = 0; i < this->layers.size(); ++i)
	{
		this->layers[i]->Serialize(config["Layer" + std::to_string(i)], weights);
	}

	std::ofstream configWriter(configfilePath);
	configWriter << config;
}

void NeuralNet::Load(std::string configfilePath, std::string weightsPath)
{
	this->layers.clear();

	std::ifstream configReader(configfilePath);
	Json::Value config;
	configReader >> config;
	std::ifstream weights(weightsPath, std::ios::binary);

	for (size_t i = 0; i < config.size(); ++i)
	{
		this->layers.push_back(ReadLayer(config["Layer" + std::to_string(i)], weights));
	}
}


Tensor OneHotEncode(uint number)
{
	cv::Mat encoded = cv::Mat::zeros(1, NUMBER_OF_CLASSES, CV_64F);
	encoded.at<double>(number) = 1;
	return encoded;
}

std::vector<Tensor> OneHotEncode(std::vector<uint> &numbers)
{
	std::vector<Tensor> result(numbers.size());
	std::transform(numbers.begin(), numbers.end(), result.begin(), [](uint value) { return OneHotEncode(value); });

	return result;
}

void NeuralNet::train(std::unique_ptr<DatasetReaderInterface>& datsetReader, int batchSize, int epoch)
{
	const uint VISUALIZATION_BATCH = 64;
	uint elapsedImageCounter = 0;
	uint visualizationCounter = 0;

	this->learningLogs.open("logs.csv");
	this->learningLogs << "accuracy, loss" << '\n';
	for (int e = 0; e < epoch; ++e)
	{
		while (datsetReader->IsDataAvailable())
		{
			auto data = datsetReader->GetDataBatch(batchSize);
			auto x = data.first;
			auto y = OneHotEncode(data.second);

			x = this->compute(x);
			y = this->backPropagate(y);

			// std::cout << "BatchNorm Gradient: \n" << TensorToCvMat(y[0]) << '\n';

			visualizationCounter += x.size();
			if (visualizationCounter >= VISUALIZATION_BATCH)
			{
				elapsedImageCounter += visualizationCounter;
				visualizationCounter = 0;

				std::cout << "Elapsed: " << elapsedImageCounter << '\n';
				y = OneHotEncode(data.second);
				this->Evaluate(x, y);
			}
		}
		datsetReader->Restart();
	}

	const uint VALIDATION_DATA_SIZE = 1000;
	std::cout << "Validation on " << VALIDATION_DATA_SIZE << '\n';

	auto data = datsetReader->GetDataBatch(VALIDATION_DATA_SIZE);
	auto x = data.first;
	auto y = OneHotEncode(data.second);

	x = this->compute(x);
	this->Evaluate(x, y);

	this->learningLogs.close();
}

Tensor NeuralNet::compute(Tensor input)
{
	/*std::cout << "Compute\n";
	auto cpuinput = std::vector<cv::Mat>{ TensorToCvMat(input) };
	std::cout << "Cuda:\n" << TensorToCvMat(input) << '\n';
	std::cout << "Cpu:\n" << cpuinput[0] << '\n';
	std::cout << "Diff:\n" << cpuinput[0] - TensorToCvMat(input) << '\n';
	std::cout << "end\n";*/
	try
	{
		for (int i = 0; i < layers.size(); ++i)
		{
			input = layers[i]->compute(input);
			/*cpuinput = cpuLayers[i]->compute(cpuinput);

			std::cout << "Cuda:\n" << TensorToCvMat(input) << '\n';
			std::cout << "Cpu:\n" << cpuinput[0] << '\n';
			std::cout << "Diff:\n" << cpuinput[0] - TensorToCvMat(input) << '\n';
			std::cout << "end\n";*/
		}
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
	return input;
}

std::vector<Tensor> NeuralNet::compute(std::vector<Tensor>& input)
{
	try
	{
		for (int i = 0; i < layers.size(); ++i)
		{
			input = layers[i]->compute(input);
		}
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
	return input;
}

Tensor NeuralNet::backPropagate(Tensor input)
{
	/*std::cout << "BackPropagate\n";
	auto cpuinput = std::vector<cv::Mat>{ TensorToCvMat(input) };*/
	try
	{
		for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
		{
			input = layers[i]->backPropagate(input);
			/*cpuinput = cpuLayers[i]->backPropagate(cpuinput);

			std::cout << "Cuda:\n" << TensorToCvMat(input) << '\n';
			std::cout << "Cpu:\n" << cpuinput[0] << '\n';
			std::cout << "Diff:\n" << cpuinput[0] - TensorToCvMat(input) << '\n';
			std::cout << "end\n";*/
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return input;
}

std::vector<Tensor>  NeuralNet::backPropagate(std::vector<Tensor>& input)
{
	try
	{
		for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
		{
			input = layers[i]->backPropagate(input);
		}
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return input;
}

double crossEntropy(Tensor result, Tensor label)
{
	Tensor mult = MatrixMult(result, 0, TransposeMatrix(label, 0), 0);
	double p = Sum(mult);
	return -std::log(p);
}


int classIndex(Tensor input)
{
	cv::Mat cvInput = TensorToCvMat(input);
	cv::Point max_loc;
	cv::minMaxLoc(cvInput, nullptr, nullptr, nullptr, &max_loc);
	return max_loc.x;
}

void NeuralNet::Evaluate(std::vector<Tensor>& predicted, std::vector<Tensor>& groundtruth)
{
	uint batchSize = predicted.size();
	uint batchAccuracy = 0;
	double loss = 0;

	for (uint i = 0; i < batchSize; ++i)
	{
		auto predictedClassIndx = classIndex(predicted[i]);
		auto groundtruthClassIndx = classIndex(groundtruth[i]);
		batchAccuracy += predictedClassIndx == groundtruthClassIndx;
		loss += crossEntropy(predicted[i], groundtruth[i]);
	}
	auto accuracy = (static_cast<double>(batchAccuracy) / static_cast<double>(batchSize));
	loss /= static_cast<double>(batchSize);
	std::cout << "Accuracy: " << accuracy << " loss: " << loss   << "\n";

	this->learningLogs << accuracy << " , " << loss << '\n';
}