#pragma once
#include "NeuralNet/Layer.h"
#include "NeuralNet/LayersTypes.h"

std::unique_ptr<LayerInterface> BuildDummyLayer(LayerTypes type);

std::unique_ptr<LayerInterface> ReadLayer(Json::Value& config, std::ifstream& weights);