#include "train_lenet.h"

#include <stdint.h>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "../datasets/data_helper.h"
#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../metrics/classification_metrics.h"
#include "../models/model_wrapper.h"
#include "../settings.h"
#include "common.h"

constexpr const char *ANSI_END = "\033[0m";
constexpr const char *ANSI_GREEN = "\033[32m";
constexpr const char *ANSI_MAGENTA = "\033[35m";
constexpr const char *ANSI_RED = "\033[31m";
constexpr const char *ANSI_YELLOW = "\033[33m";

namespace nn = torch::nn;

int lenet_loop(Settings &opts, ModelTypes modelType, DatasetTypes datasetType, bool train, bool test)
{
	// a custom dataset is a bit pointless, but it is used as "proof of concept" or if pose is ready, it is useful there
	// auto should be justified, since the datatypes are selfexplenatory and really long to typedef
	// edit: i really should have used Python

	auto /*Info*/ [trainInfo, valInfo, testInfo] = load_dataset_info(datasetType, opts.mnistOpts);
	int ret = 0;
	switch (datasetType)
	{
		case DatasetTypes::MnistType:
		{
			ret = mnist_loop(opts, modelType, trainInfo, valInfo, testInfo, train, test);
			if (ret != 0) { return ret; }
			break;
		}
		case DatasetTypes::Cifar10Type:
		{
			ret = cifar10_loop(opts, modelType, trainInfo, valInfo, testInfo, train, test);
			if (ret != 0) { return ret; }
			break;
		}
		default:
		{
			std::abort();
		}
	}
	return 0;
}

int mnist_loop(Settings &opts, ModelTypes modelType, const Info trainInfo, const Info valInfo, const Info testInfo, bool train, bool test)
{
	DatasetOpts datasetOpts = opts.mnistOpts;
	int status = 0;
	auto /*Dataset*/ [trainset, valset, testset] = make_mnist_datasets(datasetOpts, trainInfo, valInfo, testInfo);

	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(trainset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.trainBS).workers(datasetOpts.numWorkers)
	);
	auto valloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(valset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.valBS).workers(datasetOpts.numWorkers)
	);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(testset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.testBS).workers(datasetOpts.numWorkers)
	);

	if (train)
	{
		status = lenet_train(*trainloader, *valloader, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	
	if (test)
	{
		status = lenet_test(*testloader, opts, modelType, train);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	
	return 0;
}

int cifar10_loop(Settings &opts, ModelTypes modelType, const Info trainInfo, const Info valInfo, const Info testInfo, bool train, bool test)
{
	DatasetOpts datasetOpts = opts.mnistOpts;
	
	int status = 0;
	auto /*Dataset*/ [trainset, valset, testset] = make_cifar10_datasets(datasetOpts, trainInfo, valInfo, testInfo);

	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(trainset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.trainBS).workers(datasetOpts.numWorkers)
	);
	auto valloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(valset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.valBS).workers(datasetOpts.numWorkers)
	);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(testset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.testBS).workers(datasetOpts.numWorkers)
	);

	if (train)
	{
		status = lenet_train(*trainloader, *valloader, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	
	if (test)
	{
		status = lenet_test(*testloader, opts, modelType, train);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	
	return 0;
}

template<typename Dataloader>
int lenet_train(Dataloader& trainloader, Dataloader& valloader, Settings &opts, ModelTypes modelType)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	
	std::shared_ptr<ModelWrapper> modelWrapper = std::make_shared<ModelWrapper>(modelType, mnistOpts);
	modelWrapper->to(opts.dev, true);

	torch::optim::Adam optimiser(modelWrapper->parameters(), torch::optim::AdamOptions(opts.learningRate));
	torch::nn::CrossEntropyLoss lossFn;

	int ret = 0;
	float bestValLoss = std::numeric_limits<float>::infinity();
	bool valImprov = false;
	bool stop = false;
	size_t dec = 0;

	std::cout << ANSI_MAGENTA << "Starting to train " + modelWrapper->get_name() << ANSI_END << "\n";
	for (size_t epoch = 1; epoch <= opts.maxEpochs; ++epoch)
	{	
		std::cout << ANSI_GREEN << "Epoch: " << epoch << ANSI_END << std::endl;
		modelWrapper->train();

		float trainLoss = 0.0;
		int i = 0;
		for (Batch &batch : trainloader)
		{
			optimiser.zero_grad();

			torch::Tensor imgs = batch.data.to(opts.dev, true);
			torch::Tensor labels = batch.target.to(opts.dev, true).view({ -1 });

			torch::Tensor outputs = modelWrapper->forward(imgs);
			torch::Tensor loss = lossFn(outputs, labels);
			if (std::isnan(loss.template item<float>()))
			{
				std::cout << ANSI_RED << "Training is unstable, change settings" << ANSI_END << std::endl;
				return 1;
			}
			loss.backward();
			optimiser.step();

			trainLoss += loss.item<float>();
			i++;
		}

		trainLoss /= i;

		// TODO: visualise the train loss with opencv
		std::cout << ANSI_MAGENTA << "Train loss: " << trainLoss << ANSI_END << std::endl;

		if (epoch % opts.valInterval == 0)
		{
			ret = lenet_val(modelWrapper, valloader, bestValLoss, lossFn, valImprov, opts);
			if (ret == 1)
			{
				std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
				return 1;
			}
			stop = early_stopping(opts.IntervalsBeforeEarlyStopping, opts.minEpochs, epoch, valImprov);
			if (stop)
			{
				return 0;
			}
		}
	}
	
	return 0;
}

template<typename Dataloader>
int lenet_val(std::shared_ptr<ModelWrapper> modelWrapper, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	modelWrapper->eval();
	int i = 0;
	float valLoss = 0.0;
	
	torch::NoGradGuard no_grad;
	for (Batch &batch : valloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, true);
		torch::Tensor labels = batch.target.to(opts.dev, true).view({-1});

		torch::Tensor outputs = modelWrapper->forward(imgs);
		torch::Tensor loss = lossFn(outputs, labels);
		if (std::isnan(loss.template item<float>()))
		{
			std::cout << ANSI_RED << "Validation is unstable, change parameters" << ANSI_END << std::endl;
			return 1;
		}
		i++;
		valLoss += loss.item<float>();
	}
	valLoss /= i;
	
	// TODO visualise
	std::cout << ANSI_MAGENTA << "The validation loss is: " << valLoss << ANSI_END << std::endl;
	
	// < is used to avoid false improvements
	imp = false;
	if (valLoss < bestValLoss)
	{
		modelWrapper->save_weights(mnistOpts.workModel);
		bestValLoss = valLoss;
		imp = true;
		std::cout << ANSI_YELLOW << "The model improved" << ANSI_END << std::endl;
	}

	return 0;
}

template<typename Dataloader>
int lenet_test(Dataloader &testloader, Settings &opts, ModelTypes modelType, bool train)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::unique_ptr<ModelWrapper> modelWrapper = std::make_unique<ModelWrapper>(modelType, mnistOpts);
	std::cout << ANSI_MAGENTA << "Starting testing for " << modelWrapper->get_name() << ANSI_END << "\n";

	std::string fModel = train ? mnistOpts.workModel : mnistOpts.testModel;
	modelWrapper->load_weights(fModel);
	modelWrapper->to(opts.dev, true);
	modelWrapper->eval();
	MetricsContainer mc(mnistOpts.numOfClasses);

	torch::NoGradGuard no_grad;
	for (Batch &batch : testloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, true);
		torch::Tensor labels = batch.target.to(opts.dev, true).view({ -1 });
		torch::Tensor outputs = modelWrapper->forward(imgs);
		calc_cm(labels, outputs, mc);
	}
	mc.print_cm();
	mc.calc_metrics();
	mc.print_metrics();

	std::string ans;
	std::cout << "Would you like to save the model?\nType name.pth to save the model, skip by not writing in the correct format" << std::endl;
	std::getline(std::cin, ans);

	std::regex rx(R"(^\w+\.pth$)");
	if (std::regex_match(ans, rx))
	{
		std::cout << "Saving the model" << std::endl;
		modelWrapper->save_weights(ans);
	}
	else
	{
		std::cout << "Not saving the model" << std::endl;
	}
		
	return 0;
}
