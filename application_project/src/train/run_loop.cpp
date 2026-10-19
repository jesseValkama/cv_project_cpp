#include "train/run_loop.h"

#include "metrics/timer.h"
#include "train/common.h"
#include "train/test.h"
#include "train/train.h"
#include "train/validate.h"

// todo: use templates
int train::run_loop(Settings &opts, const ModelTypes modelType, const DatasetTypes datasetType, 
	const bool trainModel, const bool testModel)
{
	train::Tracker tracker = train::Tracker();
	std::vector<int> tidxs, vidxs;
	auto [trainInfo, valInfo, testInfo] = load_dataset_info(datasetType, opts.mnistOpts, tidxs, vidxs);
	int ret = 0;
	switch (datasetType)
	{
		case DatasetTypes::MNIST:
		{
			ret = mnist_helper(opts, modelType, trainInfo, valInfo, testInfo, tracker, trainModel, testModel);
			if (ret != 0) { return ret; }
			break;
		}
		case DatasetTypes::CIFAR10:
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
		Timer trainTimer = Timer();
		trainTimer.start();
		status = train_loop(*trainloader, *valloader, tracker, trainSize, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
		tracker.trainTime = trainTimer.format_time(trainTimer.checkpoint());
		std::cout << "Training took: " << tracker.trainTime << std::endl;
	}
	if (testModel)
	{
		Timer testTimer = Timer();
		testTimer.start();
		status = test_loop(*testloader, opts, modelType, datasetName, tracker, trainModel);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
		tracker.testTime = testTimer.format_time(testTimer.checkpoint());
		std::cout << "Testing took: " << tracker.testTime << std::endl;
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
		Timer trainTimer = Timer();
		trainTimer.start();
		status = train_loop(*trainloader, *valloader, tracker, trainSize, opts, modelType);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The training failed, fatal" << ANSI_END << std::endl;
			return status;
		}
		tracker.trainTime = trainTimer.format_time(trainTimer.checkpoint());
		std::cout << "Training took: " << tracker.trainTime << std::endl;
	}
	if (testModel)
	{
		Timer testTimer = Timer();
		testTimer.start();
		status = test_loop(*testloader, opts, modelType, datasetName, tracker, trainModel);
		if (status != 0)
		{
			std::cout << ANSI_RED << "The testing failed, fatal" << ANSI_END << std::endl;
			return status;
		}
		tracker.testTime = testTimer.format_time(testTimer.checkpoint());
		std::cout << "Testing took: " << tracker.testTime << std::endl;
	}
	return 0;
}