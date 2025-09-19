#include "data_helper.h"

#include <torch/torch.h>

#include <cassert>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../settings.h"

std::pair<Info, Info> split_train_val_info(Info &trainValInfo, double trainProb, int bs, std::vector<int> &tidxs, std::vector<int> &vidxs)
{
	int trainMax = 0, valMax = 0;

	Info trainInfo, valInfo;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(1, 100);
	int n = trainValInfo.size(), r = 0;

	for (int i = 0; i < n; ++i)
	{
		r = dist(gen);
		if ((float) r / 100.0 >= trainProb)
		{
			valInfo.push_back(trainValInfo[i]);
			valMax += 1;
		}
		else
		{
			trainInfo.push_back(trainValInfo[i]);
			trainMax += 1;
		}
		if ((i + 1) % bs == 0 && i > 0)
		{
			tidxs.push_back(trainMax + 1);
			vidxs.push_back(valMax + 1);
		}
	}
	trainValInfo.clear();
	return make_pair(trainInfo, valInfo);
}

std::tuple<Info, Info, Info> load_dataset_info(DatasetTypes datasetType, DatasetOpts &datasetOpts, std::vector<int> &tidxs, std::vector<int> &vidxs, double trainRatio)
{
	int ret = 0;
	Info trainValInfo, testInfo;
	switch (datasetType)
	{
		case DatasetTypes::MnistType:
		{
			ret = load_mnist_info(datasetOpts, trainValInfo, "train");
			if (ret != 0) { std::abort(); }
			ret = load_mnist_info(datasetOpts, testInfo, "test");
			if (ret != 0) { std::abort(); }
			auto [trainInfo, valInfo] = split_train_val_info(trainValInfo, trainRatio, 10000, tidxs, vidxs);
			return std::make_tuple(trainInfo, valInfo, testInfo);
		}
		case DatasetTypes::Cifar10Type:
		{
			ret = load_cifar10_info(datasetOpts, trainValInfo, "train");
			if (ret != 0) { std::abort(); }
			ret = load_cifar10_info(datasetOpts, testInfo, "test");
			if (ret != 0) { std::abort(); }
			auto [trainInfo, valInfo] = split_train_val_info(trainValInfo, trainRatio, 10000, tidxs, vidxs);
			return std::make_tuple(trainInfo, valInfo, testInfo);
		}
		default:
		{
			std::abort();
		}
	}
}

std::tuple<Mnistds, Mnistds, Mnistds> make_mnist_datasets(DatasetOpts &datasetOpts, const Info &trainInfo, const Info &valInfo, const Info &testInfo)
{
	Mnistds trainSet = MnistDataset(trainInfo, datasetOpts, "train", datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	Mnistds valSet = MnistDataset(valInfo, datasetOpts, "val", datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	Mnistds testSet = MnistDataset(testInfo, datasetOpts, "test", datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	return std::make_tuple(trainSet, valSet, testSet);
}

std::tuple<Cifar10ds, Cifar10ds, Cifar10ds> make_cifar10_datasets(DatasetOpts &datasetOpts, const Info &trainInfo, const Info &valInfo, const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs)
{
	Cifar10ds trainSet = Cifar10Dataset(trainInfo, datasetOpts, "train", tidxs, datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	Cifar10ds valSet = Cifar10Dataset(valInfo, datasetOpts, "val", vidxs, datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	Cifar10ds testSet = Cifar10Dataset(testInfo, datasetOpts, "test", std::vector<int>{10000}, datasetOpts.async)
		.map(torch::data::transforms::Normalize<>(
			datasetOpts.mean, datasetOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	return std::make_tuple(trainSet, valSet, testSet);
}
