#include "NeuralNet/CpuHelpTools.h"

#include <fstream>



void print(std::vector<cv::Mat> input)
{
	std::cout << "begin cube: \n";
	for (auto& mat : input)
	{
		std::cout << mat << '\n';
	}
	std::cout << "end cube: \n";
}

void print(std::vector<std::vector<cv::Mat>> input)
{
	std::cout << "begin tensor: \n";
	for (auto& cube : input)
	{
		print(cube);
	}
	std::cout << "end tensor: \n";
}



const std::string DATA_FOLDER = "C:/Course Work/data/";
const std::string INDEX_FILE_PREFIX = "index_";
void readDataFromFile(std::vector<std::string>& names, std::vector<int>& classes)
{
	for (int number = 0; number < 10; ++number)
	{
		std::ifstream in(DATA_FOLDER + INDEX_FILE_PREFIX + std::to_string(number) + ".txt");
		std::string imageFile;

		while (!in.eof())
		{
			in >> imageFile;

			names.push_back(imageFile);
			classes.push_back(number);
		}
	}

	//names.push_back("0.png");
	//classes.push_back(5);
}

std::vector<cv::Mat> convertToFloatValue(std::vector<cv::Mat> tensor)
{
	std::vector<cv::Mat> result(tensor.size());
	for (int i = 0; i < tensor.size(); ++i)
	{
		tensor[i].convertTo(result[i], CV_64FC1);
	}
	return result;
}

std::vector<std::vector<cv::Mat>> readImages(const std::vector<std::string>& names)
{
	std::vector<std::vector<cv::Mat>> images(names.size());

#pragma omp parallel for
	for (int i = 0; i < names.size(); ++i)
	{
		cv::Mat rgbImage = cv::imread(DATA_FOLDER + "mnist_png/" + names[i]);
		cv::Mat grayImage;
		cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

		//std::vector<cv::Mat> tensor;
		//cv::split(img, tensor);

		images[i] = convertToFloatValue({ grayImage });
	}

	return images;
}

void read(std::vector<std::vector<cv::Mat>>& inputs, std::vector<int>& classes)
{
	std::vector<std::string> names;
	readDataFromFile(names, classes);

	inputs = readImages(names);
}

void prepareData(std::vector<std::vector<cv::Mat>>& inputs)
{
	double mean = 0.f;
	int size = inputs.size() * inputs[0].size() * inputs[0][0].cols * inputs[0][0].rows;
	std::for_each(inputs.begin(), inputs.end(), [&mean, size](const std::vector<cv::Mat>& filter) mutable {
		std::for_each(filter.begin(), filter.end(), [&mean, size](const cv::Mat& mat) mutable {
			mean += sum(mat)[0] / size;
			});
		});
	std::cout << mean << std::endl;
	std::for_each(inputs.begin(), inputs.end(), [mean](std::vector<cv::Mat>& filter) {
		std::for_each(filter.begin(), filter.end(), [mean](cv::Mat& mat) {
			mat -= mean;
			});
		});

	double std = 0.f;
	std::for_each(inputs.begin(), inputs.end(), [&std, size](const std::vector<cv::Mat>& filter) mutable {
		std::for_each(filter.begin(), filter.end(), [&std, size](const cv::Mat& mat) mutable {
			std += cv::sum(mat.mul(mat))[0] / size;
			});
		});
	std::cout << std << std::endl;
	std::for_each(inputs.begin(), inputs.end(), [std](std::vector<cv::Mat>& filter) {
		std::for_each(filter.begin(), filter.end(), [std](cv::Mat& mat) {
			mat /= std;
			});
		});
}

std::vector<cv::Mat> CreateCube(int chanels, int width, int height)
{
	std::vector<cv::Mat> cube(chanels);
	for (int i = 0; i < chanels; ++i)
	{
		cube[i] = cv::Mat::zeros(height, width, CV_64FC1);
	}
	return cube;
}

std::vector<std::vector<cv::Mat>> CreateTensor(int numberOfCubes, int chanels, int width, int height)
{
	std::vector<std::vector<cv::Mat>> tensor(numberOfCubes);
	for (int i = 0; i < numberOfCubes; ++i)
	{
		tensor[i] = CreateCube(chanels, width, height);
	}
	return tensor;
}