#include "Common/HelpTools.h"

#include <exception>
#include <string>

void HandleCudaStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		throw std::exception(("Bad cuda status! Error: " + std::to_string(static_cast<int>(cudaStatus))).c_str());
	}
}

void HandleCudaRandStatus(curandStatus_t cudaStatus)
{
	if (cudaStatus != CURAND_STATUS_SUCCESS)
	{
		throw std::exception(("Bad curand status! Error: " + std::to_string(static_cast<int>(cudaStatus))).c_str());
	}
}