#include "ReLU.h"
#include "HelpTools.h"
#include "MathFunctions.h"
#include <iostream>

Tensor ReLU::compute(Tensor& input)
{
	this->inputs = input;
    return ApplyReLU(this->inputs);
}

std::vector<Tensor> ReLU::compute(std::vector<Tensor>& input)
{
    this->batchInputs = input;

    std::vector<Tensor> result;
    std::transform(input.begin(), input.end(), std::back_inserter(result),
        [this](Tensor& value) { return ApplyReLU(value); });

    return result;
}

Tensor ReLU::backPropagate(Tensor& input)
{
    return this->ComputeGradient(this->inputs, input);
}

std::vector<Tensor> ReLU::backPropagate(std::vector<Tensor>& input)
{
    if (input.size() != this->batchInputs.size())
    {
        throw std::invalid_argument("Batches must be same size!");
    }

    std::vector<Tensor> result(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        result[i] = this->ComputeGradient(this->batchInputs[i], input[i]);
    }

    return result;
}

void ReLU::print(std::ostream& out)
{
    out << "ReLU\n";
}

void ReLU::Serialize(Json::Value& config, std::ofstream&)
{
    config["name"] = "ReLU";
}

void ReLU::DeSerialize(Json::Value&, std::ifstream&)
{
}

__global__ void ComputeDerivative(double* input, double* dx, uint size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * blockDim.x * gridDim.x;

    while (tid < size) {
        if (input[tid] <= 0)
        {
            dx[tid] = 0;
        }
        tid += blockDim.y * blockDim.x * gridDim.x;
    }
}

Tensor ReLU::ComputeGradient(Tensor& layerInput, Tensor& gradient)
{
    Tensor dx(gradient);

    const uint BLOCK_SIZE = 16;
    uint sqrtDataSize = static_cast<uint>(std::sqrt(dx.dataSize) + 1);
    uint gridRows = (sqrtDataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint gridCols = (sqrtDataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridCols, gridRows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    ComputeDerivative<<<dimGrid, dimBlock>>>(layerInput.data, dx.data, dx.dataSize);

    return dx;
}