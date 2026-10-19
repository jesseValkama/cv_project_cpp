#ifndef SETTINGS_H
#define SETTINGS_H

#include <optional>
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include <yaml-cpp/yaml.h>

constexpr const char *ANSI_END = "\033[0m";
constexpr const char *ANSI_GREEN = "\033[32m";
constexpr const char *ANSI_MAGENTA = "\033[35m";
constexpr const char *ANSI_RED = "\033[31m";
constexpr const char *ANSI_YELLOW = "\033[33m";

// TODO: move
enum DatasetTypes
{
	DS_INIT = 0,
	MNIST = 1,
	CIFAR10 = 2
};

enum ModelTypes
{
	MOD_INIT = 0,
	LENET = 1,
	RESNET = 2
};

enum VisualiserTasks
{
	SKIP = 0,
    IDX = 1,
    GRADCAM = 2
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
	std::vector<std::string> labels = {};
	int64_t numOfChannels = 0;
	
	std::vector<double> mean = {};
	std::vector<double> stdev = {};
	
	DatasetOpts(YAML::Node &yaml, ModelTypes modelType = ModelTypes::MOD_INIT, DatasetTypes datasetType = DatasetTypes::DS_INIT);
	void assign_from_mnist(ModelTypes modelType, YAML::Node &yaml);
	void assign_from_cifar10(ModelTypes modelType, YAML::Node &yaml);
};

struct Settings
{
	YAML::Node yaml = YAML::LoadFile("/home/jesse/code/cv_project_cpp/application_project/settings.yaml");
	DatasetOpts mnistOpts = DatasetOpts(yaml);
	std::string databasePath = "";

	torch::Device dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	c10::DeviceType devType = torch::cuda::is_available() ? c10::DeviceType::CUDA : c10::DeviceType::CPU; 
	c10::ScalarType ampDtype = torch::kFloat16;
	size_t minEpochs = 0;
	size_t maxEpochs = 0;
	size_t valInterval = 0;
	float learningRate = 0.0;
	float weightDecay = 0.0;
	size_t IntervalsBeforeEarlyStopping = 0; // unnecessary for lenet, but could be useful for a more complex model
	bool amp = false;
	bool gradScaler = false;

	int warmupLen = 0;
	int plateauWait = 0;

	Settings(DatasetTypes datasetType, ModelTypes modelType);
};

#endif // SETTINGS_H