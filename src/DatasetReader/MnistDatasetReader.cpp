#include "DatasetReader/MnistDatasetReader.h"
#include <fstream>
#include <algorithm>

const std::string INDEX_FILE_PREFIX = "/index_";
const std::string MNIST_FOLDER = "mnist_png";

std::vector<std::pair<unsigned int, std::string>> ReadMnist(const std::string path)
{
	std::vector<std::pair<unsigned int, std::string>> datasetFileNames;

#pragma omp parallel for
	for (int number = 0; number < 10; ++number)
	{
		std::ifstream in(path + INDEX_FILE_PREFIX + std::to_string(number) + ".txt");
		std::string imageFile;
		std::vector<std::pair<unsigned int, std::string>> temp;

		while (in.is_open() && !in.eof())
		{
			in >> imageFile;

			temp.push_back({ number, path + "/" + MNIST_FOLDER + "/" + imageFile });
		}

#pragma omp atomic
		std::copy(temp.begin(), temp.end(), std::back_inserter(datasetFileNames));
	}

	return datasetFileNames;
}