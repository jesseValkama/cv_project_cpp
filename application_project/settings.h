#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdint.h>
#include <string>

#include <torch/torch.h>

struct MnistOpts
{
	std::string fimgname = "D:/datasets/mnist/train-images.idx3-ubyte";
	std::string flabelname = "D:/datasets/mnist/train-labels.idx1-ubyte";
	std::string savepath = "D:/self-studies/application_project/application_project/weights/model.pth";
	torch::Device dev = torch::kCUDA;

	int imgsz = 28;
	size_t trainBS = 32;
	size_t valBS = 32;
	size_t testBS = 32;
	size_t numWorkers = 4;
	/*size_t iters = 10;
	size_t interval = 64;*/

};

struct Settings
{
	MnistOpts mnistOpts;
	size_t maxEpochs = 30;
	size_t valInterval = 6;
	float learningRate = 0.005;
	int numOfChannels = 10;
	size_t IntervalsBeforeEarlyStopping = 5;
	bool automatedMixedPrecision = false;
};

#endif