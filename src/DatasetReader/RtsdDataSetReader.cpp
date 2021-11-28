#include "DatasetReader/RtsdDatasetReader.h"
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>

const std::string TRAIN_CSV_FILE_NAME = "/gt_train.csv";
const std::string RTSD_TRAIN_FOLDER = "train";

std::vector<std::pair<unsigned int, std::string>> ReadRTSD(const std::string path)
{
	std::vector<std::pair<unsigned int, std::string>> datasetFileNames;
	std::ifstream csvFile(path + TRAIN_CSV_FILE_NAME);

	std::string current_line;
	// read columns names
	if (!getline(csvFile, current_line))
	{
		return {};
	}

	while (getline(csvFile, current_line)) {
		// Now inside each line we need to seperate the cols
		std::pair<unsigned int, std::string> values;
		std::stringstream temp(current_line);
		std::string single_value;
		if (!getline(temp, single_value, ','))
		{
			return {};
		}
		values.second = path + '/' + RTSD_TRAIN_FOLDER + '/' + single_value;

		if (!getline(temp, single_value, ','))
		{
			return {};
		}
		values.first = atoi(single_value.c_str());


		// add the row to the complete data vector
		datasetFileNames.push_back(values);
	}

	return datasetFileNames;
}