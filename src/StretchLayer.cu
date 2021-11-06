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

Json::Value StretchLayer::Serialize()
{
	Json::Value json;
	json["StretchLayer"]["width"] = this->width;
	json["StretchLayer"]["height"] = this->height;
	json["StretchLayer"]["channels"] = this->channels;
	json["StretchLayer"]["four"] = this->four;

	return json;
}

void StretchLayer::DeSerialize(Json::Value json)
{
	this->width = json["StretchLayer"]["width"].asInt();
	this->height = json["StretchLayer"]["height"].asInt();
	this->channels = json["StretchLayer"]["channels"].asInt();
	this->four = json["StretchLayer"]["four"].asInt();
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