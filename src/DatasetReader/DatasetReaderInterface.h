#pragma once
#include "Common/Tensor.h"

class DatasetReaderInterface
{
public:
	virtual std::pair<Tensor, uint> GetData() = 0;
	virtual std::pair<std::vector<Tensor>, std::vector<uint>> GetDataBatch(uint batchSize) = 0;
	virtual bool IsDataAvailable() = 0;
	virtual void Restart() = 0;
};