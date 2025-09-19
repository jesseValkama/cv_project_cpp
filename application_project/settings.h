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