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
	std::string savepath = "D:/self-studies/application_project/application_project/weights";
	std::string inferenceModel = "lenet_inference"; // you can leave out the formatting like .pth, it is automatically added
	std::string testModel = "lenet_inference";
	std::string workModel = "working";
	int imgsz = 28;
	int imgresz = 224; // 32 lenet, 224 resnet
	size_t trainBS = 64; // 128 lenet, 64 resnet
	size_t valBS = 64;
	size_t testBS = 64;
	size_t numWorkers = 4;
	int64_t numOfClasses = 10;
	int64_t numOfChannels = 1;
	
	// source for the hardcoded mean and stdev values: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python 
	float mean = 0.1307;
	float stdev = 0.3081;
};

struct Settings
{
	MnistOpts mnistOpts;
	torch::Device dev = torch::kCUDA;
	size_t minEpochs = 5;
	size_t maxEpochs = 8;
	size_t valInterval = 2;
	float learningRate = 0.005;
	size_t IntervalsBeforeEarlyStopping = 1; // unnecessary for lenet, but could be useful for a more complex model
	bool automatedMixedPrecision = false; // NOT IN USE YET
};

#endif