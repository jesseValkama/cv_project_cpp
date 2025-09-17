#ifndef SETTINGS_H
#define SETTINGS_H

#include <optional>
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

// this is here to avoid headache with circular imports
enum DatasetTypes
{
	DatasetInitType = 0,
	MnistType = 1,
	Cifar10Type = 2
};

enum ModelTypes
{
	ModelInitType = 0,
	LeNetType = 1,
	ResNetType = 2
};

struct Cifar10Opts
{
	// download the dataset from https://www.cs.toronto.edu/%7Ekriz/cifar.html
	std::string trainBatch1 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin";
	std::string trainBatch2 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin";
	std::string trainBatch3 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin";
	std::string trainBatch4 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin";
	std::string trainBatch5 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin";

	std::string meta = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt";
	std::string testBatch1 = "D:/datasets/cifar-10-binary/cifar-10-batches-bin/test_batch.bin";

	std::vector<std::string> fTrain = { trainBatch1 };
	std::vector<std::string> fTest = { testBatch1 };

	std::string inferenceDataPath = "D:/datasets/inference_imgs";
	std::string savepath = "D:/self-studies/application_project/application_project/weights";
	std::string inferenceModel = "lenet_inference"; // you can leave out the formatting like .pth, it is automatically added
	std::string testModel = "lenet_inference";
	std::string workModel = "working";

	int imgsz = 32;
	int imgresz = 224; // 32 lenet, 224 resnet
	size_t trainBS = 128; // 128 lenet, 64 resnet
	size_t valBS = 128;
	size_t testBS = 128;
	size_t numWorkers = 6;
	int64_t numOfClasses = 10;
	int64_t numOfChannels = 3;

	// credit: https://www.kaggle.com/code/abdelrahmanhesham601/resnet-18-fine-tuning-on-cifar-10
	std::vector<double> mean = {0.4914, 0.4822, 0.4465};
	std::vector<double> stdev = { 0.247, 0.243, 0.261 };
};

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
	int imgresz = 32; // 32 lenet, 224 resnet
	size_t trainBS = 128; // 128 lenet, 64 resnet
	size_t valBS = 128;
	size_t testBS = 128;
	size_t numWorkers = 6;
	int64_t numOfClasses = 10;
	int64_t numOfChannels = 1;
	
	// source for the hardcoded mean and stdev values: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python 
	std::vector<double> mean = { 0.1307 };
	std::vector<double> stdev = { 0.3081 };
};

struct DatasetOpts
{
	// do not touch this class, use the subclasses
	std::string fTrainImgs = "";
	std::string fTrainLabels = "";
	std::string fTestImgs = "";
	std::string fTestLabels = "";

	std::string meta = "";
	std::vector<std::string> fTrain = {};
	std::vector<std::string> fTest = {};

	std::string inferenceDataPath = "";
	std::string savepath = "";
	std::string inferenceModel = "";
	std::string testModel = "";
	std::string workModel = "";

	int imgsz = 0;
	int imgresz = 0;
	size_t trainBS = 0;
	size_t valBS = 0;
	size_t testBS = 0;
	size_t numWorkers = 0;
	bool async = false;
	int64_t numOfClasses = 0;
	int64_t numOfChannels = 0;
	
	std::vector<double> mean = {};
	std::vector<double> stdev = {};
	
	DatasetOpts(YAML::Node &yaml, ModelTypes modelType = ModelTypes::ModelInitType, DatasetTypes datasetType = DatasetTypes::DatasetInitType);
	void assign_from_mnist(ModelTypes modelType, YAML::Node &yaml);
	void assign_from_cifar10(ModelTypes modelType, YAML::Node &yaml);
};

struct Settings
{
	YAML::Node yaml = YAML::LoadFile("D:/self-studies/application_project/application_project/settings.yaml");
	DatasetOpts mnistOpts = DatasetOpts(yaml);

	torch::Device dev = torch::kCUDA;
	size_t minEpochs = 0;
	size_t maxEpochs = 0;
	size_t valInterval = 0;
	float learningRate = 0.0;
	float weightDecay = 0.0;
	size_t IntervalsBeforeEarlyStopping = 1; // unnecessary for lenet, but could be useful for a more complex model
	bool automatedMixedPrecision = false; // NOT IN USE YET

	Settings(DatasetTypes datasetType, ModelTypes modelType);
};

#endif