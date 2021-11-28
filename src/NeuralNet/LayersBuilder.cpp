#include "NeuralNet/LayersBuilder.h"
#include "NeuralNet/StretchLayer.h"
#include "NeuralNet/FullyConnected.h"
#include "NeuralNet/ReLU.h"
#include "NeuralNet/SoftMax.h"
#include "NeuralNet/MaxPooling.h"
#include "NeuralNet/BatchNormalization.h"
#include "NeuralNet/ConvLayer.h"
#include <iostream>


std::unique_ptr<LayerInterface> BuildStrecthLayer()
{
	return std::make_unique<StretchLayer>(0, 0);
}

std::unique_ptr<LayerInterface> BuildFullyConnected()
{
	try
	{
		return std::make_unique<FullyConnected>(0, 0, 0);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}
	return nullptr;
}

std::unique_ptr<LayerInterface> BuildReLU()
{
	return std::make_unique<ReLU>();
}

std::unique_ptr<LayerInterface> BuildSoftMax()
{
	return std::make_unique<SoftMax>();
}

std::unique_ptr<LayerInterface> BuildMaxPooling()
{
	return std::make_unique<MaxPooling>(0, 0);
}

std::unique_ptr<LayerInterface> BuildBatchNormalization()
{
	return std::make_unique<BatchNormalization>(0, 0, 0);
}

std::unique_ptr<LayerInterface> BuildConvLayer()
{
	return std::make_unique<ConvLayer>(1, 1, 1, 1, 1);
}

std::unique_ptr<LayerInterface> BuildDummyLayer(LayerTypes type)
{
	switch (type)
	{
	case StretchLayerType:
		return BuildStrecthLayer();
		break;
	case FullyConnectedType:
		return BuildFullyConnected();
		break;
	case ReLUType:
		return BuildReLU();
		break;
	case SoftMaxType:
		return BuildSoftMax();
		break;
	case MaxPoolingType:
		return BuildMaxPooling();
		break;
	case BatchNormalizationType:
		return BuildBatchNormalization();
		break;
	case ConvLayerType:
		return BuildConvLayer();
		break;
	default:
		return nullptr;
		break;
	}
}

LayerTypes RecognizeLayerType(const char* name)
{
	if (strcmp(name, "StretchLayer") == 0)
	{
		return StretchLayerType;
	}
	else if (strcmp(name, "FullyConnected") == 0)
	{
		return FullyConnectedType;
	}
	else if (strcmp(name, "ReLU") == 0)
	{
		return ReLUType;
	}
	else if (strcmp(name, "SoftMax") == 0)
	{
		return SoftMaxType;
	}
	else if (strcmp(name, "MaxPooling") == 0)
	{
		return MaxPoolingType;
	}
	else if (strcmp(name, "BatchNormalization") == 0)
	{
		return BatchNormalizationType;
	}
	else if (strcmp(name, "ConvLayer") == 0)
	{
		return ConvLayerType;
	}
	else
	{
		return UnRecognnizedType;
	}
}

std::unique_ptr<LayerInterface> ReadLayer(Json::Value& config, std::ifstream& weights)
{
	auto type = RecognizeLayerType(config["name"].asCString());
	std::unique_ptr<LayerInterface> layer = BuildDummyLayer(type);

	if (layer != nullptr)
	{
		layer->DeSerialize(config, weights);
	}

	return layer;
}