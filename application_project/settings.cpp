#include "settings.h"

#include <yaml-cpp/yaml.h>

DatasetOpts::DatasetOpts(YAML::Node &yaml, ModelTypes modelType, DatasetTypes datasetType)
{
	switch (datasetType)
	{
		case DatasetTypes::MnistType:
			assign_from_mnist(modelType, yaml);
			break;

		case DatasetTypes::Cifar10Type:
			assign_from_cifar10(modelType, yaml);
			break;
	}
}

void DatasetOpts::assign_from_mnist(ModelTypes modelType, YAML::Node &yaml)
{
	fTrainImgs = yaml["mnist_data_paths"]["train_data"]["train_imgs"].as<std::string>();
	fTrainLabels = yaml["mnist_data_paths"]["train_data"]["train_labels"].as<std::string>();
	fTestImgs = yaml["mnist_data_paths"]["test_data"]["test_imgs"].as<std::string>();
	fTestLabels = yaml["mnist_data_paths"]["test_data"]["test_labels"].as<std::string>();
	inferenceDataPath = yaml["mnist_data_paths"]["inference_data"]["inference_imgs"].as<std::string>();

	savepath = yaml["mnist_model_paths"]["save_path"].as<std::string>();
	inferenceModel = yaml["mnist_model_paths"]["inference_name"].as<std::string>();
	testModel = yaml["mnist_model_paths"]["test_name"].as<std::string>();
	workModel = yaml["mnist_model_paths"]["work_name"].as<std::string>();

	imgsz = yaml["mnist_train_settings"]["dataset_info"]["image_size"].as<int>();
	switch (modelType)
	{
		case ModelTypes::LeNetType:
		{
			imgresz = yaml["lenet_imgresz"].as<int>();
			break;
		}
		case ModelTypes::ResNetType:
		{
			imgresz = yaml["resnet_imgresz"].as<int>();
			break;
		}
	}
	trainBS = yaml["mnist_train_settings"]["batch_sizes"]["train"].as<size_t>();
	valBS = yaml["mnist_train_settings"]["batch_sizes"]["val"].as<size_t>();
	testBS = yaml["mnist_train_settings"]["batch_sizes"]["test"].as<size_t>();
	numWorkers = yaml["mnist_train_settings"]["multiprocessing"]["num_workers"].as<size_t>();
	async = yaml["mnist_train_settings"]["multiprocessing"]["async"].as<bool>();
	numOfClasses = yaml["mnist_train_settings"]["dataset_info"]["num_classes"].as<int64_t>();
	numOfChannels = yaml["mnist_train_settings"]["dataset_info"]["num_channels"].as<int64_t>();
	
	YAML::Node means = yaml["mnist_train_settings"]["dataset_info"]["mean"];
	for (size_t i = 0; i < means.size(); i++)
	{
		mean.push_back(means[i].as<double>());
	}
	YAML::Node stdevs = yaml["mnist_train_settings"]["dataset_info"]["standard_deviation"];
	for (size_t i = 0; i < stdevs.size(); i++)
	{
		stdev.push_back(stdevs[i].as<double>());
	}
}

void DatasetOpts::assign_from_cifar10(ModelTypes modelType, YAML::Node &yaml)
{
	meta = yaml["cifar10_data_paths"]["meta_data"]["meta_path"].as<std::string>();
	YAML::Node trainPaths = yaml["cifar10_data_paths"]["train_paths"];
	for (size_t i = 0; i < trainPaths.size(); i++)
	{
		fTrain.push_back(trainPaths[i].as<std::string>());
	}
	YAML::Node testPaths = yaml["cifar10_data_paths"]["test_paths"];
	for (size_t i = 0; i < testPaths.size(); i++)
	{
		fTest.push_back(testPaths[i].as<std::string>());
	}
	inferenceDataPath = yaml["cifar10_data_paths"]["inference_data"]["inference_path"].as<std::string>();

	savepath = yaml["cifar10_model_paths"]["save_path"].as<std::string>();
	inferenceModel = yaml["cifar10_model_paths"]["inference_name"].as<std::string>();
	testModel = yaml["cifar10_model_paths"]["test_name"].as<std::string>();
	workModel = yaml["cifar10_model_paths"]["work_name"].as<std::string>();

	imgsz = yaml["cifar10_train_settings"]["dataset_info"]["image_size"].as<int>();
	switch (modelType)
	{
		case ModelTypes::LeNetType:
		{
			imgresz = yaml["lenet_imgresz"].as<int>();
			break;
		}
		case ModelTypes::ResNetType:
		{
			imgresz = yaml["resnet_imgresz"].as<int>();
			break;
		}
	}
	trainBS = yaml["cifar10_train_settings"]["batch_sizes"]["train"].as<size_t>();
	valBS = yaml["cifar10_train_settings"]["batch_sizes"]["val"].as<size_t>();
	testBS = yaml["cifar10_train_settings"]["batch_sizes"]["test"].as<size_t>();
	numWorkers = yaml["cifar10_train_settings"]["multiprocessing"]["num_workers"].as<size_t>();
	async = yaml["cifar10_train_settings"]["multiprocessing"]["async"].as<bool>();
	numOfClasses = yaml["cifar10_train_settings"]["dataset_info"]["num_classes"].as<int64_t>();
	numOfChannels = yaml["cifar10_train_settings"]["dataset_info"]["num_channels"].as<int64_t>();
	
	YAML::Node means = yaml["cifar10_train_settings"]["dataset_info"]["mean"];
	for (size_t i = 0; i < means.size(); i++)
	{
		mean.push_back(means[i].as<double>());
	}
	YAML::Node stdevs = yaml["cifar10_train_settings"]["dataset_info"]["standard_deviation"];
	for (size_t i = 0; i < stdevs.size(); i++)
	{
		stdev.push_back(stdevs[i].as<double>());
	}
}

Settings::Settings(DatasetTypes datasetType, ModelTypes modelType)
{
	minEpochs = yaml["general_settings"]["loop"]["min_epochs"].as<size_t>();
	maxEpochs = yaml["general_settings"]["loop"]["max_epochs"].as<size_t>();
	valInterval = yaml["general_settings"]["loop"]["validation_interval"].as<size_t>();
	learningRate = yaml["general_settings"]["optimiser"]["learning_rate"].as<float>();
	weightDecay = yaml["general_settings"]["optimiser"]["weight_decay"].as<float>();
	IntervalsBeforeEarlyStopping = yaml["general_settings"]["loop"]["early_stop_counter"].as<size_t>();
	automatedMixedPrecision = yaml["general_settings"]["loop"]["amp"].as<bool>(); // NOT IN USE YET

	DatasetOpts datasetOpts(yaml, modelType, datasetType);
	mnistOpts = datasetOpts;
}
