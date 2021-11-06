#pragma once
#include <opencv2/core.hpp>
#include <json/json.h>
#include "HelpTools.h"

struct Tensor
{
	Tensor();
	Tensor(uint width, uint height, uint channels = 1, uint four = 1);
	Tensor(const cv::Mat& value);
	Tensor(Json::Value json);
	Tensor(const Tensor& value);
	Tensor(Tensor&& value);
	virtual ~Tensor();

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
Json::Value TensorToJson(Tensor& value);

Tensor GenerateUniformDistributionTensor(uint width, uint height, uint channels, uint four);
Tensor GenerateNormalDistributionTensor(uint width, uint height, uint channels, uint four, double mean, double stddev);

Tensor GenerateFilledTensor(uint width, uint height, uint channels, uint four, double value);
Tensor JoinTensors(std::vector<Tensor>& value);
std::vector<Tensor> SplitTensors(Tensor& value);

void PrintTensor(Tensor& value);