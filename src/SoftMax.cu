#include "SoftMax.h"
#include "HelpTools.h"
#include "MathFunctions.h"

Tensor SoftMax::compute(Tensor& input)
{
    this->output = this->ComputeSoftMax(input);
    return this->output;
}

std::vector<Tensor> SoftMax::compute(std::vector<Tensor>& input)
{
    this->batchOutputs.clear();
    std::transform(input.begin(), input.end(), std::back_inserter(batchOutputs),
        [this](Tensor& value) { return this->ComputeSoftMax(value); });

    return this->batchOutputs;
}

Tensor SoftMax::backPropagate(Tensor& input)
{
    return this->ComputeError(this->output, input);
}

std::vector<Tensor> SoftMax::backPropagate(std::vector<Tensor>& input)
{
    if (input.size() != this->batchOutputs.size())
    {
        throw std::invalid_argument("Batches must be same size!");
    }

    std::vector<Tensor> result(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        result[i] = this->ComputeError(this->batchOutputs[i], input[i]);
    }

    return result;
}

void SoftMax::print(std::ostream& out)
{
    out << "SoftMax\n";
}

void SoftMax::Serialize(Json::Value& config, std::ofstream&)
{
    config["SoftMax"] = "";
}

void SoftMax::DeSerialize(Json::Value&, std::ifstream&)
{
}

Tensor SoftMax::ComputeSoftMax(Tensor& input)
{
    Tensor result = ApplyExp(input);
    double norm = Sum(result);
    result = result * (1. / norm);

    return result;
}

Tensor SoftMax::ComputeError(Tensor& softmaxOutput, Tensor& gradient)
{
    return softmaxOutput - gradient;
}