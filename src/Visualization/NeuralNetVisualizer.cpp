#include "Visualization/NeuralNetVisualizer.h"
#include <opencv2/highgui.hpp>
#include <functional>

const std::string WINDOW_NAME = "RTSD Classifier";
const size_t DATASET_IMAGE_SIZE = 48;
const size_t VIEWER_SIZE = 7 * DATASET_IMAGE_SIZE;

NeuralNetVisualizer::NeuralNetVisualizer(std::string configfilePath, std::string weightsPath,
	std::unique_ptr<DatasetReaderInterface>&& datsetReader)
{
	data.net = std::make_unique<NeuralNet>(configfilePath, weightsPath);
	data.datsetReader = std::move(datsetReader);
	
	data.image = cv::Mat3b(VIEWER_SIZE, VIEWER_SIZE, cv::Vec3b(255,255, 255));
	data.dataRect = cv::Rect(3*DATASET_IMAGE_SIZE, 3*DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
	data.button = cv::Rect(2*DATASET_IMAGE_SIZE, 5 * DATASET_IMAGE_SIZE, 3 * DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);

	cv::rectangle(data.image, data.button, cv::Scalar(0, 255, 0), 2);
	cv::putText(data.image, "Next image", cv::Point(data.button.x, data.button.y + DATASET_IMAGE_SIZE/2),
		cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
	
	data.predictedPoint = cv::Point(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
	data.truePoint = cv::Point(DATASET_IMAGE_SIZE + 3* DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
}

cv::Mat ConvertToImage(Tensor tensor)
{
	cv::Mat cvImage;
	auto convertedImage = TensorToCvMat(tensor);

	if (cvImage.channels() == 1)
	{
		cv::Mat temp;
		convertedImage.convertTo(temp, CV_8U);
		cv::cvtColor(temp, cvImage, cv::COLOR_GRAY2BGR);
	}
	else
	{
		convertedImage.convertTo(cvImage, CV_8UC3);
	}

	return cvImage;
}

int getIndex(Tensor input)
{
	cv::Mat cvInput = TensorToCvMat(input);
	cv::Point max_loc;
	cv::minMaxLoc(cvInput, nullptr, nullptr, nullptr, &max_loc);
	return max_loc.x;
}

void Callback(int event, int x, int y, int flags, void* userdata)
{
	NeuralNetVisualizer::VisualData* data = reinterpret_cast<NeuralNetVisualizer::VisualData*>(userdata);

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		if (data->button.contains(cv::Point(x, y)))
		{
			std::cout << "Clicked!" << std::endl;
			auto& [tensor, id] = data->datsetReader->GetData();
			auto predictedId = getIndex(data->net->compute(tensor));

			data->image.setTo(cv::Vec3b(255, 255, 255));
			cv::putText(data->image, "predicted: " + std::to_string(predictedId + 1),
				data->predictedPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			cv::putText(data->image, "expected: " + std::to_string(id),
				data->truePoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

			auto dataImage = ConvertToImage(tensor);
			dataImage.copyTo(data->image(data->dataRect));

			cv::rectangle(data->image, data->button, cv::Scalar(0, 255, 0), 2);
			cv::putText(data->image, "Next image", cv::Point(data->button.x, data->button.y + DATASET_IMAGE_SIZE / 2),
				cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
		}
	}

	cv::imshow(WINDOW_NAME, data->image);
}

void NeuralNetVisualizer::RunVisualization()
{
	// Setup callback function
	cv::namedWindow(WINDOW_NAME);
	cv::setMouseCallback(WINDOW_NAME, Callback, reinterpret_cast<void*>(&data));

	cv::imshow(WINDOW_NAME, data.image);
	cv::waitKey();
}