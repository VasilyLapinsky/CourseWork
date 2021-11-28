#pragma once

#include "cuda_runtime.h"
#include "curand.h"

void HandleCudaStatus(cudaError_t cudaStatus);
void HandleCudaRandStatus(curandStatus_t cudaStatus);