#pragma once
#include "NeuralNet/Layer.h"
#include "DatasetReader/DatasetReaderInterface.h"
#include "NeuralNet/CpuLayers.h"

class NeuralNet
{
public:
	NeuralNet();
	NeuralNet(std::string configfilePath, std::string weightsPath);

public:
	void addLayer(std::shared_ptr<LayerInterface> layer);
	void Save(std::string configfilePath, std::string weightsPath);
	void Load(std::string configfilePath, std::string weightsPath);

public:
	void train(std::unique_ptr<DatasetReaderInterface> &datsetReader, int batchSize, int epoch);

public:
	Tensor compute(Tensor input);
	std::vector<Tensor> compute(std::vector<Tensor>& input);
	Tensor backPropagate(Tensor input);
	std::vector<Tensor> backPropagate(std::vector<Tensor>& input);

private:
	void Evaluate(std::vector<Tensor>& predicted, std::vector<Tensor>& groundtruth);

private:
	std::vector<std::shared_ptr<LayerInterface>> layers;
	std::vector<std::shared_ptr<cpu::Layer>> cpuLayers;
};