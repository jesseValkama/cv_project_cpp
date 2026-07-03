#include "functions/train_lenet.h"

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

#include "datasets/data_helper.h"
#include "datasets/loader_funcs.h"
#include "datasets/mnist.h"
#include "metrics/classification_metrics.h"
#include "models/model_wrapper.h"
#include "settings.h"
#include "optims/scheds/warmup.h"
#include "functions/common.h"
#include "functions/experiments_database.h"

constexpr const char *ANSI_END = "\033[0m";
constexpr const char *ANSI_GREEN = "\033[32m";
constexpr const char *ANSI_MAGENTA = "\033[35m";
constexpr const char *ANSI_RED = "\033[31m";
constexpr const char *ANSI_YELLOW = "\033[33m";

namespace nn = torch::nn;

int train::run_loop(Settings &opts, const ModelTypes modelType, const DatasetTypes datasetType, 
	const bool trainModel, const bool testModel)
{
	train::Tracker tracker = train::Tracker();
	std::vector<int> tidxs, vidxs;
	auto [trainInfo, valInfo, testInfo] = load_dataset_info(datasetType, opts.mnistOpts, tidxs, vidxs);
	int ret = 0;
	switch (datasetType)
	{
		case DatasetTypes::MnistType:
		{
			ret = mnist_helper(opts, modelType, trainInfo, valInfo, testInfo, tracker, trainModel, testModel);
			if (ret != 0) { return ret; }
			break;
		}
		case DatasetTypes::Cifar10Type:
		{
			ret = cifar10_helper(opts, modelType, trainInfo, valInfo, testInfo, tidxs, vidxs, tracker, trainModel, testModel);
			if (ret != 0) { return ret; }
			break;
		}
		default:
		{
			return 1;
		}
	}
	return 0;
}

// this is terrible code having almost similar fns but libtorch uses unique ptrs and the inputs come from the cli

int mnist_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, train::Tracker &tracker, const bool trainModel, const bool testModel)
{
	const std::string datasetName = "Mnist";
	std::cout << ANSI_MAGENTA << "Starting to load the " << datasetName << " dataset" << ANSI_END << std::endl;
	DatasetOpts datasetOpts = opts.mnistOpts;
	int status = 0;
	auto [trainset, valset, testset] = make_mnist_datasets(datasetOpts, trainInfo, valInfo, testInfo);

	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(trainset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.trainBS).workers(datasetOpts.numWorkers)
	);
	auto valloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
	(
		std::move(valset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.valBS).workers(datasetOpts.numWorkers)
	);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
	(
		std::move(testset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.testBS).workers(datasetOpts.numWorkers)
	);
	const size_t trainSize = trainset.size().value();

	if (trainModel)
	{
		status = train_loop(*trainloader, *valloader, tracker, trainSize, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	if (testModel)
	{
		status = test_loop(*testloader, opts, modelType, datasetName, tracker, trainModel);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	return 0;
}

int cifar10_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs, 
	train::Tracker &tracker, const bool trainModel, const bool testModel)
{
	const std::string datasetName = "Cifar10";
	std::cout << ANSI_MAGENTA << "Starting to load the " << datasetName << " dataset" << ANSI_END << std::endl;
	DatasetOpts datasetOpts = opts.mnistOpts;
	int status = 0;
	auto [trainset, valset, testset] = make_cifar10_datasets(datasetOpts, trainInfo, valInfo, testInfo, tidxs, vidxs);

	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
	(
		std::move(trainset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.trainBS).workers(datasetOpts.numWorkers)
	);
	auto valloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
	(
		std::move(valset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.valBS).workers(datasetOpts.numWorkers)
	);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
	(
		std::move(testset),
		torch::data::DataLoaderOptions().batch_size(datasetOpts.testBS).workers(datasetOpts.numWorkers)
	);
	int trainSize = trainset.size().value();

	if (trainModel)
	{
		status = train_loop(*trainloader, *valloader, tracker, trainSize, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	if (testModel)
	{
		status = test_loop(*testloader, opts, modelType, datasetName, tracker, trainModel);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
	}
	return 0;
}


template<typename Randomloader, typename Sequentialloader>
int train_loop(Randomloader &trainloader, Sequentialloader &valloader, train::Tracker &tracker, const size_t trainSize,
	Settings &opts, ModelTypes modelType)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::shared_ptr<ModelWrapper> modelWrapper = std::make_shared<ModelWrapper>(modelType, mnistOpts);
	modelWrapper->to(opts.dev);
	torch::optim::AdamW optimiser(modelWrapper->parameters(), torch::optim::AdamWOptions(opts.learningRate).weight_decay(opts.weightDecay));
	torch::optim::ReduceLROnPlateauScheduler plateau(optimiser, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min, 0.1, opts.plateauWait);
	optims::sched::warmupLR warmup(optimiser, opts.learningRate, opts.warmupLen * trainSize / mnistOpts.trainBS);
	torch::nn::CrossEntropyLoss lossFn;

	int ret = 0;
	int i = 0;
	bool valImprov = false;
	bool stop = false;
	size_t dec = 0;

	std::cout << ANSI_MAGENTA << "Starting to train " + modelWrapper->get_name() << ANSI_END << "\n";
	for (tracker.epoch = 1; tracker.epoch <= opts.maxEpochs; ++tracker.epoch)
	{	
		std::cout << ANSI_GREEN << "Epoch: " << tracker.epoch << ANSI_END << std::endl;
		modelWrapper->train();

		tracker.trainLoss = 0.0;
		i = 0;
		for (Batch &batch : trainloader)
		{
			optimiser.zero_grad();

			torch::Tensor imgs = batch.data.to(opts.dev, mnistOpts.async);
			torch::Tensor labels = batch.target.to(opts.dev, mnistOpts.async).view({ -1 });

			torch::Tensor outputs = modelWrapper->forward(imgs);
			torch::Tensor loss = lossFn(outputs, labels);
			if (std::isnan(loss.template item<double>()))
			{
				std::cout << ANSI_RED << "Training is unstable, change settings" << ANSI_END << std::endl;
				return 1;
			}
			loss.backward();
			optimiser.step();

			tracker.trainLoss += loss.item<double>();
			++i;
			warmup.step();
		}
		tracker.trainLoss /= i;

		// TODO: visualise the train loss with opencv
		std::cout << ANSI_MAGENTA << "Train loss: " << tracker.trainLoss << ANSI_END << std::endl;

		tracker.valLoss = 0.0;
		if (tracker.epoch % opts.valInterval == 0)
		{
			ret = val_loop(modelWrapper, valloader, tracker, lossFn, valImprov, opts);
			if (ret == 1)
			{
				std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
				return 1;
			}
			stop = early_stopping(opts.IntervalsBeforeEarlyStopping, opts.minEpochs, tracker.epoch, valImprov);
			if (stop)
			{
				return 0;
			}
		}
		plateau.step(tracker.valLoss);
	}
	return 0;
}

template<typename Sequentialloader>
int val_loop(std::shared_ptr<ModelWrapper> modelWrapper, Sequentialloader &valloader, train::Tracker &tracker, 
	nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	modelWrapper->eval();
	int i = 0;
	
	torch::NoGradGuard no_grad;
	for (Batch &batch : valloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, mnistOpts.async);
		torch::Tensor labels = batch.target.to(opts.dev, mnistOpts.async).view({-1});

		torch::Tensor outputs = modelWrapper->forward(imgs);
		torch::Tensor loss = lossFn(outputs, labels);
		if (std::isnan(loss.template item<double>()))
		{
			std::cout << ANSI_RED << "Validation is unstable, change parameters" << ANSI_END << std::endl;
			return 1;
		}
		i++;
		tracker.valLoss += loss.item<double>();
	}
	tracker.valLoss /= i;
	
	// TODO visualise
	std::cout << ANSI_MAGENTA << "The validation loss is: " << tracker.valLoss << ANSI_END << std::endl;
	
	imp = false;
	if (tracker.valLoss < tracker.bestValLoss)
	{
		modelWrapper->save_weights(mnistOpts.workModel);
		tracker.bestValLoss = tracker.valLoss;
		imp = true;
		std::cout << ANSI_YELLOW << "The model improved" << ANSI_END << std::endl;
	}
	return 0;
}

template<typename Sequentialloader>
int test_loop(Sequentialloader &testloader, Settings &opts, const ModelTypes modelType, const std::string &datasetName,
	 train::Tracker &tracker, const bool useTrainedModel)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::unique_ptr<ModelWrapper> modelWrapper = std::make_unique<ModelWrapper>(modelType, mnistOpts);
	std::string modelName = modelWrapper->get_name();
	std::cout << ANSI_MAGENTA << "Starting testing for " << modelName << ANSI_END << "\n";

	std::string fModel = useTrainedModel ? mnistOpts.workModel : mnistOpts.testModel;
	modelWrapper->load_weights(fModel);
	modelWrapper->to(opts.dev);
	modelWrapper->eval();
	MetricsContainer mc(mnistOpts.numOfClasses);

	torch::NoGradGuard no_grad;
	for (Batch &batch : testloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, mnistOpts.async);
		torch::Tensor labels = batch.target.to(opts.dev, mnistOpts.async).view({ -1 });
		torch::Tensor outputs = modelWrapper->forward(imgs);
		calc_cm(labels, outputs, mc);
	}
	mc.print_cm();
	mc.calc_metrics();
	mc.print_metrics();
	mc.print_metrics(-2);
	AvgMetrics metrics = mc.get_metrics();

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
		ans = "not_saved";
		std::cout << "Not saving the model" << std::endl;
	}
	int ret = insert_experiments("/home/jesse/code/cv_project_cpp/application_project/test.db", 
		"experimentName", modelName.c_str(), datasetName.c_str(), "trainTime", tracker.epoch, ans.c_str(), metrics.at("recall"), 
									metrics.at("precision"), metrics.at("accuracy"), "optimiserName");
	return ret;
}
