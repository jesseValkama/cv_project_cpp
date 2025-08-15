#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdint.h>
#include <string>

#include <torch/torch.h>

struct Settings
{
	size_t maxEpochs = 30;
	size_t valInterval = 4;
	float learningRate = 0.001;
	int numOfChannels = 10;
	size_t IntervalsBeforeEarlyStopping = 5;
	bool automatedMixedPrecision = false;
};

struct MnistOptions
{
	int imgSize = 32;
	size_t trainBS = 8;
	size_t valBS = 8;
	size_t testBS = 8;
	size_t iters = 10;
	size_t interval = 64;

	std::string datasetPath = "";
	std::string infoFilePath = "";
	torch::Device dev = torch::kCUDA;
};

#endif