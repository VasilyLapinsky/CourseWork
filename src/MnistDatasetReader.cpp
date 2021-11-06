#include "MnistDatasetReader.h"
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

const std::string INDEX_FILE_PREFIX = "index_";
const std::string MNIST_FOLDER = "mnist_png";

void ShuffleData(std::vector<std::pair<uint, std::string>>& data)
{
	size_t size = data.size();
	for (size_t i = size - 1; i > 0; --i) {
		size_t newIndex = std::rand() % (i + 1);
		std::swap(data[i], data[newIndex]);
	}
}

std::vector<std::pair<uint, std::string>> ReadImageFileNames(const std::string path, bool shufle)
{
	std::vector<std::pair<uint, std::string>> datasetFileNames;

#pragma omp parallel for
	for (int number = 0; number < 10; ++number)
	{
		std::ifstream in(path + INDEX_FILE_PREFIX + std::to_string(number) + ".txt");
		std::string imageFile;
		std::vector<std::pair<uint, std::string>> temp;

		while (!in.eof())
		{
			in >> imageFile;

			temp.push_back({ number, path + "/" + MNIST_FOLDER + "/" + imageFile });
		}

#pragma omp atomic
		std::copy(temp.begin(), temp.end(), std::back_inserter(datasetFileNames));
	}

	if (shufle)
	{
		ShuffleData(datasetFileNames);
	}
	return datasetFileNames;
}

MnistDatasetReader::MnistDatasetReader(const std::string path, bool shufle)
	:datasetFileNames{ ReadImageFileNames(path, shufle) }
{
	this->datasetIterator = datasetFileNames.begin();
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

std::pair<Tensor, uint> MnistDatasetReader::GetData()
{
	if (!this->IsDataAvailable())
	{
		return { Tensor(), 0 };
	}

	auto file = *this->datasetIterator;
	this->datasetIterator++;

	return { ReadImage(file.second), file.first };
}

std::pair<std::vector<Tensor>, std::vector<uint>> MnistDatasetReader::GetDataBatch(uint batchSize)
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

	for (int i = 0; i < resultBatchSize; ++i)
	{
		auto file = *this->datasetIterator;
		this->datasetIterator++;

		numbers[i] = file.first;
		images[i] = ReadImage(file.second);
	}

	return { images, numbers };
}

bool MnistDatasetReader::IsDataAvailable()
{
	return this->datasetIterator != datasetFileNames.end();
}

void MnistDatasetReader::Restart()
{
	this->datasetIterator = datasetFileNames.begin();
}