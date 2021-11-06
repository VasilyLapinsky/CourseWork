#include "HelpTools.h"

#include <exception>

void HandleCudaStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		throw std::exception("Bad cuda status! Error: " + cudaStatus);
	}
}

void HandleCudaRandStatus(curandStatus_t cudaStatus)
{
	if (cudaStatus != CURAND_STATUS_SUCCESS)
	{
		throw std::exception("Bad curand status! Error: " + cudaStatus);
	}
}