#pragma once
#include "NeuralNet/NeuralNet.h"
#include <opencv2/highgui.hpp>

class NeuralNetVisualizer
{
public:
	struct VisualData
	{
		std::unique_ptr<NeuralNet> net;
		std::unique_ptr<DatasetReaderInterface> datsetReader;

		cv::Mat3b image;
		cv::Rect dataRect;
		cv::Rect button;

		cv::Point predictedPoint;
		cv::Point truePoint;
	};

public:
	NeuralNetVisualizer(std::string configfilePath, std::string weightsPath, 
						std::unique_ptr<DatasetReaderInterface>&& datsetReader);

public:
	void RunVisualization();

private:
	VisualData data;
};