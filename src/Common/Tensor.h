#pragma once
#include <opencv2/core.hpp>
#include <json/json.h>
#include <fstream>
#include "Common/HelpTools.h"

struct Tensor
{
	Tensor();
	Tensor(uint width, uint height, uint channels = 1, uint four = 1);
	Tensor(const cv::Mat& value);
	Tensor(const Tensor& value);
	Tensor(Json::Value& config, std::ifstream& binaryData);
	Tensor(Tensor&& value);
	virtual ~Tensor();

	void Serrialize(Json::Value& config, std::ofstream& binaryData);

	void operator+=(double value);
	Tensor operator+(double value) const;

	void operator*=(double value);
	Tensor operator*(double value) const;

	void operator+=(Tensor& value);
	Tensor operator+(Tensor& value) const;

	void operator-=(Tensor& value);
	Tensor operator-(Tensor& value) const;

	Tensor operator=(const Tensor& value);

	uint width;
	uint height;
	uint channels;
	uint four;
	uint dataSize;
	double* data;
};

cv::Mat TensorToCvMat(Tensor& value);

Tensor GenerateUniformDistributionTensor(uint width, uint height, uint channels, uint four);
Tensor GenerateNormalDistributionTensor(uint width, uint height, uint channels, uint four, double mean, double stddev);

Tensor GenerateFilledTensor(uint width, uint height, uint channels, uint four, double value);
Tensor JoinTensors(std::vector<Tensor>& value);
std::vector<Tensor> SplitTensors(Tensor& value);

void PrintTensor(Tensor& value);