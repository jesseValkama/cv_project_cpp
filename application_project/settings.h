#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdint.h>
#include <string>

#include <torch/torch.h>

struct MnistOpts
{
	std::string fimgname = "D:/datasets/mnist/t10k-images.idx3-ubyte";
	std::string flabelname = "D:/datasets/mnist/t10k-labels.idx1-ubyte";
	torch::Device dev = torch::kCUDA;

	int imgsz = 28;
	size_t trainBS = 8;
	size_t valBS = 8;
	size_t testBS = 8;
	size_t iters = 10;
	size_t interval = 64;

};

struct Settings
{
	MnistOpts mnistOpts;
	size_t maxEpochs = 30;
	size_t valInterval = 4;
	float learningRate = 0.001;
	int numOfChannels = 10;
	size_t IntervalsBeforeEarlyStopping = 5;
	bool automatedMixedPrecision = false;
};

#endif