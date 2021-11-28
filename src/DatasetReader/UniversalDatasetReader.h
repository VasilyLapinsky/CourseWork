#pragma once
#include "DatasetReader/DatasetReaderInterface.h"

class UniversalDatasetReader : public DatasetReaderInterface
{
public:
	UniversalDatasetReader(std::vector<std::pair<uint, std::string>> &datasetFileNames, bool shufle = true);

public:
	std::pair<Tensor, uint> GetData() override;
	std::pair<std::vector<Tensor>, std::vector<uint>> GetDataBatch(uint batchSize) override;
	bool IsDataAvailable() override;
	void Restart() override;

private:
	std::vector<std::pair<uint, std::string>> datasetFileNames;
	std::vector<std::pair<uint, std::string>>::iterator datasetIterator;
};