#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdint.h>
#include <string>

#include <torch/torch.h>

struct MnistOpts
{
	// i downloaded the dataset from kaggle
	std::string fTrainImgs = "D:/datasets/mnist/train-images.idx3-ubyte";
	std::string fTrainLabels = "D:/datasets/mnist/train-labels.idx1-ubyte";
	std::string fTestImgs = "D:/datasets/mnist/t10k-images.idx3-ubyte";
	std::string fTestLabels = "D:/datasets/mnist/t10k-labels.idx1-ubyte";

	std::string inferenceDataPath = "D:/datasets/inference_imgs";
	std::string savepath = "D:/self-studies/application_project/application_project/weights/model.pth"; // todo: fix the name

	int imgsz = 28;
	int imgresz = 32;
	size_t trainBS = 128;
	size_t valBS = 128;
	size_t testBS = 128;
	size_t numWorkers = 4;
	int numOfChannels = 10;
	
	// source for the hardcoded mean and stdev values: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python 
	float mean = 0.1307;
	float stdev = 0.3081;
};

struct Settings
{
	MnistOpts mnistOpts;
	torch::Device dev = torch::kCUDA;
	size_t maxEpochs = 16;
	size_t valInterval = 2;
	float learningRate = 0.005;
	size_t IntervalsBeforeEarlyStopping = 2;
	bool automatedMixedPrecision = false;
};

#endif