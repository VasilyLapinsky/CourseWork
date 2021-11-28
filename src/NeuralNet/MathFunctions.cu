#include "NeuralNet/MathFunctions.h"
#include "Common/HelpTools.h"
#include <iostream>

const uint MAX_THREADS = 32;
const uint MAX_SOLO_THREADS = 1024;

__global__ void mult(double* left, double* right, double* result, uint size) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        result[tid] = left[tid] * right[tid];
        tid += blockDim.x * gridDim.x;
    }
}

Tensor PerElementMult(Tensor& left, Tensor& right)
{
    if (left.four != right.four
        || left.channels != right.channels
        || left.width != right.width
        || left.height != right.height)
    {
        throw std::exception("Invalid size!");
    }

    Tensor result(left.width, left.height, left.channels, left.four);

    dim3 grids((left.dataSize + MAX_THREADS - 1) / MAX_THREADS);
    dim3 threads(MAX_THREADS);

    mult<<<grids, threads>>>(left.data, right.data, result.data, result.dataSize);

    HandleCudaStatus(cudaGetLastError());

    return result;
}

__global__ void div(double* left, double* right, double* result, uint size) {
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        result[tid] = left[tid] / right[tid];
        tid += blockDim.x * gridDim.x;
    }
}

Tensor PerElementDiv(Tensor& left, Tensor& right)
{
    if (left.four != right.four
        || left.channels != right.channels
        || left.width != right.width
        || left.height != right.height)
    {
        throw std::exception("Invalid size!");
    }

    Tensor result(left.width, left.height, left.channels, left.four);

    dim3 grids((left.dataSize + MAX_THREADS - 1) / MAX_THREADS);
    dim3 threads(MAX_THREADS);

    div<<<grids, threads>>>(left.data, right.data, result.data, result.dataSize);

    HandleCudaStatus(cudaGetLastError());

    return result;
}

__global__ void MatrixMult(double* left, double* right, double* result, uint m, uint n, uint k)
{
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (col < k && row < m)
    {
        for (uint i = 0; i < n; i++)
        {
            sum += left[row * n + i] * right[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

Tensor MatrixMult(Tensor& left, uint channelLeft, Tensor& right, uint channelRight)
{
    if (left.width != right.height)
    {
        throw std::exception(("left.width: " + std::to_string(left.width) + " != right.height: " + std::to_string(right.height)).c_str());
    }

    uint m = left.height;
    uint n = left.width; 
    uint k = right.width; 

    Tensor result(k, m);
    
    const uint BLOCK_SIZE = 16;
    uint gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridCols, gridRows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    double* pointerToLeftData = left.data + channelLeft * left.width * left.height;
    double* pointerToRightData = right.data + channelRight * right.width * right.height;
    MatrixMult<<<dimGrid, dimBlock>>>(pointerToLeftData, pointerToRightData, result.data, m, n, k);

    return result;
}

__global__ void TransposeMatrix(double* value, double* result, uint rows, uint cols)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        uint pos = idy * cols + idx;
        uint trans_pos = idx * rows + idy;
        result[trans_pos] = value[pos];
    }
}

Tensor TransposeMatrix(Tensor& value, uint channel)
{
    Tensor result(value.height, value.width);

    const uint BLOCK_SIZE = 16;
    uint gridRows = (value.height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint gridCols = (value.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridCols, gridRows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    double* pointerToData = value.data + channel * value.width * value.height;
    TransposeMatrix<<<dimGrid, dimBlock>>>(pointerToData, result.data, value.height, value.width);

    return result;
}


__global__ void reduce(double* data, double* result, const uint dataSize) {

    extern __shared__ double sharedData[];

    // each thread loads one element from global to shared mem
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[threadIdx.x] = 0;

    while (i < dataSize)
    {
        sharedData[threadIdx.x] += data[i];
        i += blockDim.x * gridDim.x;
    }

    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __syncthreads();
    // do reduction in shared mem
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;
        if (index + s < blockDim.x)
        {
            sharedData[index] += sharedData[index + s];
        }
        
        __syncthreads();
    }
    
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        result[blockIdx.x] = sharedData[0];
    }
}

double Sum(Tensor& value)
{
    double hostTemp[1];
    double* deviceTemp;
    HandleCudaStatus(cudaMalloc((void**)&deviceTemp, sizeof(double)));

    uint threads = std::min(value.dataSize, MAX_SOLO_THREADS);
    reduce << <1, threads, threads * sizeof(double) >> > (value.data, deviceTemp, value.dataSize);
    HandleCudaStatus(cudaGetLastError());

    HandleCudaStatus(cudaMemcpy((void*)hostTemp, (void*)deviceTemp, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(deviceTemp);
    return hostTemp[0];
}

Tensor SumForChannels(Tensor& value)
{
    Tensor result(value.channels, 1);
    const uint valueChannelSize = value.width * value.height;
    uint threads = std::min(valueChannelSize, MAX_SOLO_THREADS);
    for (uint i = 0; i < value.channels; ++i)
    {
        reduce<<<1, threads, threads * sizeof(double) >>>(value.data + i * valueChannelSize, result.data+i, valueChannelSize);
        HandleCudaStatus(cudaGetLastError());
    }
    return result;
}

__global__ void ReduceCols(double* value, double* result, uint rows, uint cols)
{
    uint tid = blockIdx.x;
    if (tid < rows)
    {
        result[tid] = 0;
        for (uint column = 0; column < cols; ++column)
        {
            result[tid] += value[cols * tid + column];
        }
    }
}

Tensor ReduceCols(Tensor& value)
{
    Tensor result(1, value.height);
    ReduceCols<<<value.height, 1>>>(value.data, result.data, value.height, value.width);
    return result;
}

__device__ double relu(double value)
{
    return value > 0 ? value : 0;
}

__global__ void apply_relu(double* data, double* result, uint size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = idx; i < size; i += gridDim.x * blockDim.x) {
        result[i] = relu(data[i]);
    }
}

Tensor ApplyReLU(Tensor& value)
{
    Tensor result(value.width, value.height, value.channels, value.four);

    const uint BLOCK_SIZE = 256;
    const uint NUM_BLOCKS = (result.dataSize + BLOCK_SIZE - 1) / (result.dataSize);
    apply_relu<<<NUM_BLOCKS, BLOCK_SIZE >>>(value.data, result.data, result.dataSize);

    return result;
}

__global__ void apply_exp(double* data, double* result, uint size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = idx; i < size; i += gridDim.x * blockDim.x) {
        result[i] = exp(data[i]);
    }
}

Tensor ApplyExp(Tensor& value)
{
    Tensor result(value.width, value.height, value.channels, value.four);

    const uint BLOCK_SIZE = 256;
    const uint NUM_BLOCKS = (result.dataSize + BLOCK_SIZE - 1) / (result.dataSize);
    apply_exp<<<NUM_BLOCKS, BLOCK_SIZE>>>(value.data, result.data, result.dataSize);

    return result;
}

__global__ void apply_sqrt(double* data, double* result, uint size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = idx; i < size; i += gridDim.x * blockDim.x) {
        result[i] = sqrt(data[i]);
    }
}

Tensor ApplySqrt(Tensor& value)
{
    Tensor result(value.width, value.height, value.channels, value.four);

    const uint BLOCK_SIZE = 256;
    const uint NUM_BLOCKS = (result.dataSize + BLOCK_SIZE - 1) / (result.dataSize);
    apply_sqrt<<<NUM_BLOCKS, BLOCK_SIZE>>>(value.data, result.data, result.dataSize);

    return result;
}