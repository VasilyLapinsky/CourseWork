#pragma once
#include "NeuralNet/CpuHelpTools.h"

namespace cpu
{
	class Layer
	{
	public:
		virtual std::vector<cv::Mat> compute(std::vector<cv::Mat>& input) = 0;
		virtual std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& input) = 0;
	};

	class SoftMax : public Layer
	{

	private:
		cv::Mat output;
	public:

		std::vector<cv::Mat> compute(std::vector<cv::Mat>& input);
		std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& rightAnswer);
	};

	class ReLU : public Layer
	{
	private:

		std::vector<cv::Mat> inputs;

	public:

		std::vector<cv::Mat> compute(std::vector<cv::Mat>& input) override;
		std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& input) override;
	};

	class FullyConnected : public Layer
	{
	public:
		cv::Mat weights;
		cv::Mat bias;
		cv::Mat inputs;

		double lambda;

	public:
		FullyConnected(int numInputs, int numOutput, float lambda);

		std::vector<cv::Mat> compute(std::vector<cv::Mat>& input);
		std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& input);
	};

	class StretchLayer : public Layer
	{
	private:
		int chanels, width, height;
	public:

		std::vector<cv::Mat> compute(std::vector<cv::Mat>& input) override;
		std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& input) override;
	};

	class ConvLayer : public Layer
	{
	public:
		int inputChanels, kernelSize, numFilters;

		std::vector<cv::Mat> inputs;
		std::vector<std::vector<cv::Mat>> weights;
		std::vector<std::vector<cv::Mat>> dwMemmory;
		std::vector<double> bias;

		int stride, padding;
		double lambda;

	public:
		ConvLayer(int inputChanels, int numFilters, int kernelSize, int stride, float lambda, int padding = 0);

		std::vector<cv::Mat> compute(std::vector<cv::Mat>& input) override;
		std::vector<cv::Mat> backPropagate(std::vector<cv::Mat>& input) override;
	};
}