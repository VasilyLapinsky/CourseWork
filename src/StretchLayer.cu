#include "StretchLayer.h"

const char* StretchLayerConfigNodeName = "StretchLayer";

StretchLayer::StretchLayer(uint width, uint height, uint channels, uint four)
	: width{ width }
	, height{ height }
	, channels{ channels }
	, four{ four }
{
}

Tensor StretchLayer::compute(Tensor& input)
{
	return this->ReShapeIntoRow(input);
}

std::vector<Tensor> StretchLayer::compute(std::vector<Tensor>& input)
{
	std::vector<Tensor> result;

	std::transform(input.begin(), input.end(), std::back_inserter(result), 
		[this](Tensor& value) { return this->ReShapeIntoRow(value); });

	return result;
}

Tensor StretchLayer::backPropagate(Tensor& input)
{
	return this->ReShapeBackward(input);
}

std::vector<Tensor> StretchLayer::backPropagate(std::vector<Tensor>& input)
{
	std::vector<Tensor> result;

	std::transform(input.begin(), input.end(), std::back_inserter(result),
		[this](Tensor& value) { return this->ReShapeBackward(value); });

	return result;
}

void StretchLayer::print(std::ostream& out)
{
	out << "StretchLayer\n";
	out << "Channels: " << this->channels << " Width: " << this->width << " Height: " << this->height <<'\n';
}

void StretchLayer::Serialize(Json::Value& config, std::ofstream&)
{
	config[StretchLayerConfigNodeName]["width"] = this->width;
	config[StretchLayerConfigNodeName]["height"] = this->height;
	config[StretchLayerConfigNodeName]["channels"] = this->channels;
	config[StretchLayerConfigNodeName]["four"] = this->four;
}

void StretchLayer::DeSerialize(Json::Value& config, std::ifstream&)
{
	this->width = config[StretchLayerConfigNodeName]["width"].asInt();
	this->height = config[StretchLayerConfigNodeName]["height"].asInt();
	this->channels = config[StretchLayerConfigNodeName]["channels"].asInt();
	this->four = config[StretchLayerConfigNodeName]["four"].asInt();
}

Tensor StretchLayer::ReShapeIntoRow(Tensor& input)
{
	Tensor result(input);
	result.width = result.dataSize;
	result.height = 1;
	result.channels = 1;
	result.four = 1;

	return result;
}

Tensor StretchLayer::ReShapeBackward(Tensor& input)
{
	if (input.dataSize != this->four * this->channels * this->width * this->height)
	{
		throw std::exception("Invalid input size!");
	}

	Tensor result(input);
	result.width = this->width;
	result.height = this->height;
	result.channels = this->channels;
	result.four = this->four;

	return result;
}