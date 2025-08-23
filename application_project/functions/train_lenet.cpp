#include "train_lenet.h"

#include <stdint.h>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp> // used for debugging, remember to remove at some point

#include "../datasets/mnist.h"
#include "../datasets/loader_funcs.h"
#include "../models/lenet.h"
#include "../settings.h"
#include "common.h"

namespace nn = torch::nn;

int lenet_loop(Settings &opts, bool train, bool test)
{
	std::cout << "\033[35m" << "Starting to train lenet" << "\033[0m" << "\n";
	MnistOpts mnistOpts = opts.mnistOpts;
	
	// TODO: split train into train and val
	// a custom dataset is a bit pointless, but it is used as "proof of concept" or if pose is ready, it is useful there
	Info trainValInfo, testInfo;
	int status = 0;
	status = load_mnist_info(opts.mnistOpts, trainValInfo, "train");
	if (status != 0)
	{
		return status;
	}
	status = load_mnist_info(opts.mnistOpts, testInfo, "test");
	if (status != 0)
	{
		return status;
	}
	auto [trainInfo, valInfo] = split_train_val_info(trainValInfo, 0.85);
	
	auto trainset = MnistDataset(trainInfo, mnistOpts, "train")
		.map(torch::data::transforms::Normalize<>(
			{mnistOpts.mean}, {mnistOpts.stdev}))
		.map(torch::data::transforms::Stack<>());
	auto valset = MnistDataset(valInfo, mnistOpts, "val")
		.map(torch::data::transforms::Normalize<>(
			{mnistOpts.mean}, {mnistOpts.stdev}))
		.map(torch::data::transforms::Stack<>());
	auto testset = MnistDataset(testInfo, mnistOpts, "test")
		.map(torch::data::transforms::Normalize<>(
			{mnistOpts.mean}, {mnistOpts.stdev}))
		.map(torch::data::transforms::Stack<>());
	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(trainset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.trainBS).workers(mnistOpts.numWorkers)
		);
	auto valloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(valset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.valBS).workers(mnistOpts.numWorkers)
		);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(testset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.testBS).workers(mnistOpts.numWorkers)
		);
	
	if (train) 
	{
		status = lenet_train(*trainloader, *valloader, opts);
		if (status == 1)
		{
			return status;
		}
	}
	
	if (test)
	{
		status = lenet_test(*testloader, opts);
		if (status == 1)
		{
			std::cout << "\033[31m" << "The testing failed, fatal" << "\033[0m" << std::endl;
			return status;
		}
	}
	
	return 0;
}

template<typename Dataloader>
int lenet_train(Dataloader& trainloader, Dataloader& valloader, Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgresz);
	model->to(opts.dev);

	torch::optim::Adam optimiser(model->parameters(), torch::optim::AdamOptions(opts.learningRate));
	torch::nn::CrossEntropyLoss lossFn;

	int ret = 0;
	float bestValLoss = std::numeric_limits<float>::infinity();
	bool valImprov = false;
	bool stop = false;
	size_t dec = 0;

	for (size_t epoch = 1; epoch <= opts.maxEpochs; ++epoch)
	{	
		std::cout << "\033[32m" << "Epoch: " << epoch << "\033[0m" << std::endl;
		model->train();

		float trainLoss = 0.0;
		int i = 0;
		for (Batch &batch : trainloader)
		{
			torch::Tensor imgs = batch.data.to(opts.dev);
			torch::Tensor labels = batch.target.to(opts.dev).view({ -1 });
			
			optimiser.zero_grad();

			torch::Tensor outputs = model->forward(imgs);
			torch::Tensor loss = lossFn(outputs, labels);
			if (std::isnan(loss.template item<float>()))
			{
				std::cout << "\033[31m" << "Training is unstable, change settings" << "\033[0m" << std::endl;
				return 1;
			}
			loss.backward();
			optimiser.step();

			trainLoss += loss.item<float>();
			i++;
		}

		trainLoss /= i;

		// TODO: visualise the train loss with opencv
		std::cout << "\033[35m" << "Train loss: " << trainLoss << "\033[0m" << std::endl;

		if (epoch % opts.valInterval == 0)
		{
			ret = lenet_val(model, valloader, bestValLoss, lossFn, valImprov, opts);
			if (ret == 1)
			{
				std::cout << "\033[31m" << "The training failed, fatal" << "\033[0m" << std::endl;
				return 1;
			}
			stop = early_stopping(opts.IntervalsBeforeEarlyStopping, valImprov);
			if (stop)
			{
				return 0;
			}
		}
	}
	
	return 0;
}

template<typename Dataloader>
int lenet_val(LeNet &model, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	model->eval();
	int i = 0;
	float valLoss = 0.0;
	
	torch::NoGradGuard no_grad;
	for (Batch &batch : valloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev);
		torch::Tensor labels = batch.target.to(opts.dev).view({-1});

		torch::Tensor outputs = model->forward(imgs);
		torch::Tensor loss = lossFn(outputs, labels);
		if (std::isnan(loss.template item<float>()))
		{
			std::cout << "\033[31m" << "Validation is unstable, change parameters" << "\033[0m" << std::endl;
			return 1;
		}
		i++;
		valLoss += loss.item<float>();
	}
	valLoss /= i;
	
	// TODO visualise
	std::cout << "\033[35m" << "The validation loss is: " << valLoss << "\033[0m" << std::endl;
	
	// < is used to avoid false improvements
	imp = false;
	if (valLoss < bestValLoss)
	{
		// TODO: error handling if the path doesn't exist
		torch::save(model, mnistOpts.savepath + "/working.pth");
		bestValLoss = valLoss;
		imp = true;
		std::cout << "\033[33m" << "The model improved" << "\033[0m" << std::endl;
	}

	return 0;
}

template<typename Dataloader>
int lenet_test(Dataloader &testloader, Settings &opts)
{
	std::cout << "\033[35m" << "Starting testing" << "\033[0m" << "\n";
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgresz);
	torch::load(model, mnistOpts.savepath + "/working.pth");
	model->to(opts.dev);
	model->eval();
	MetricsContainer mc = create_mc(mnistOpts.numOfChannels);

	torch::NoGradGuard no_grad;
	for (Batch &batch : testloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev);
		torch::Tensor labels = batch.target.to(opts.dev).view({ -1 });
		torch::Tensor outputs = model->forward(imgs);
		calc_cm(labels, outputs, mc);
	}
	mc.print_cm();
	mc.calc_metrics(mnistOpts.numOfChannels);
	mc.print_metrics();

	std::string ans;
	std::cout << "Would you like to save the model?\nType name.pth to save the model, skip by writing no" << std::endl;
	std::getline(std::cin, ans);

	std::regex rx(R"(^\w+\.pth$)");
	if (std::regex_match(ans, rx))
	{
		std::cout << "Saving the model" << std::endl;
		torch::save(model, mnistOpts.savepath + "/" + ans);
	}
	else
	{
		std::cout << "Not saving the model" << std::endl;
	}
		
	return 0;
}
