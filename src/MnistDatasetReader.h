#pragma once
#include "DatasetReaderInterface.h"

class MnistDatasetReader : public DatasetReaderInterface
{
public:
	MnistDatasetReader(const std::string path, bool shufle = true);

public:
	std::pair<Tensor, uint> GetData() override;
	std::pair<std::vector<Tensor>, std::vector<uint>> GetDataBatch(uint batchSize) override;
	bool IsDataAvailable() override;
	void Restart() override;

private:
	std::vector<std::pair<uint, std::string>> datasetFileNames;
	std::vector<std::pair<uint, std::string>>::iterator datasetIterator;
};