#include "DatasetReader/UniversalDatasetReader.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

void ShuffleData(std::vector<std::pair<uint, std::string>>& data)
{
	size_t size = data.size();
	for (size_t i = size - 1; i > 0; --i) {
		size_t newIndex = std::rand() % (i + 1);
		std::swap(data[i], data[newIndex]);
	}
}

UniversalDatasetReader::UniversalDatasetReader(std::vector<std::pair<uint, std::string>> &inputDatasetFileNames, bool shufle)
	: datasetFileNames{inputDatasetFileNames}
{
	if (shufle)
	{
		ShuffleData(this->datasetFileNames);
	}

	this->datasetIterator = this->datasetFileNames.begin();
}

Tensor ReadImage(const std::string path)
{
	cv::Mat rgbImage = cv::imread(path);
	cv::Mat grayImage;
	cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat imageDoubleDataType;
	grayImage.convertTo(imageDoubleDataType, CV_64F);

	Tensor image(imageDoubleDataType);
	return image;
}

std::pair<Tensor, uint> UniversalDatasetReader::GetData()
{
	if (!this->IsDataAvailable())
	{
		return { Tensor(), 0 };
	}

	auto file = *this->datasetIterator;
	this->datasetIterator++;

	return { ReadImage(file.second), file.first };
}

std::pair<std::vector<Tensor>, std::vector<uint>> UniversalDatasetReader::GetDataBatch(uint batchSize)
{
	std::vector<Tensor> images;
	std::vector<uint> numbers;

	if (!this->IsDataAvailable())
	{
		return { images, numbers };
	}

	uint resultBatchSize = std::min(static_cast<uint>(std::distance(this->datasetIterator, datasetFileNames.end())), batchSize);
	images.resize(resultBatchSize);
	numbers.resize(resultBatchSize);

	for (uint i = 0; i < resultBatchSize; ++i)
	{
		auto file = *this->datasetIterator;
		this->datasetIterator++;

		numbers[i] = file.first;
		images[i] = ReadImage(file.second);
	}

	return { images, numbers };
}

bool UniversalDatasetReader::IsDataAvailable()
{
	return this->datasetIterator != datasetFileNames.end();
}

void UniversalDatasetReader::Restart()
{
	this->datasetIterator = datasetFileNames.begin();
}