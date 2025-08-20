#include "train_lenet.h"

#include <stdint.h>
#include <iostream>
#include <limits>
#include <memory>
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

int lenet_loop(Settings &opts)
{
	std::cout << "Starting to train lenet" << "\n";
	MnistOpts mnistOpts = opts.mnistOpts;
	
	// TODO: split train into train and val
	// a custom dataset is a bit pointless, but it is used as "proof of concept" or if pose is ready, it is useful there
	Info trainInfo, testInfo;
	int status = 0;
	status = load_mnist_info(opts.mnistOpts, trainInfo, "train");
	if (status != 0)
	{
		return status;
	}
	status = load_mnist_info(opts.mnistOpts, testInfo, "test");
	if (status != 0)
	{
		return status;
	}

	// source for the hardcoded mean and stdev values: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python 
	auto trainset = MnistDataset(trainInfo, mnistOpts, "train")
		.map(torch::data::transforms::Normalize<>(
			{0.1307}, {0.3081}))
		.map(torch::data::transforms::Stack<>());
	auto testset = MnistDataset(testInfo, mnistOpts, "test")
		.map(torch::data::transforms::Normalize<>(
			{ 0.1307 }, { 0.3081 }))
		.map(torch::data::transforms::Stack<>());
	auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(trainset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.trainBS).workers(mnistOpts.numWorkers)
		);
	auto testloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(testset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.testBS).workers(mnistOpts.numWorkers)
		);
	
	status = lenet_train(*trainloader, *trainloader, opts);
	if (status == 1)
	{
		return status;
	}

	status = lenet_test(*testloader, opts);
	if (status == 1)
	{
		std::cout << "The testing failed, fatal" << std::endl;
		return status;
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
		std::cout << "Epoch: " << epoch << std::endl;
		model->train();

		float trainLoss = 0.0;
		int i = 0;
		for (Batch &batch : trainloader)
		{
			torch::Tensor imgs = batch.data.to(opts.dev);
			torch::Tensor labels = batch.target.to(opts.dev).view({ -1 });
			
			// TODO: make a dbg func for Tensortomat
			/*torch::Tensor dbg = imgs[0].detach().cpu();
			std::cout << dbg.sizes() << std::endl;
			dbg = dbg.squeeze();
			dbg = dbg.mul(0.3081).add(0.1307);
			dbg = dbg.mul(255).clamp(0, 255).to(torch::kUInt8);
			int h = dbg.size(0), w = dbg.size(1);
			cv::Mat cvdbg(h, w, CV_8UC1, dbg.data_ptr());
			std::cout << cvdbg << std::endl;*/

			optimiser.zero_grad();

			torch::Tensor outputs = model->forward(imgs);
			torch::Tensor loss = lossFn(outputs, labels);
			if (std::isnan(loss.template item<float>()))
			{
				std::cout << "Training is unstable, change settings" << std::endl;
				return 1;
			}
			loss.backward();
			optimiser.step();

			trainLoss += loss.item<float>();
			i++;
		}

		trainLoss /= i;

		// TODO: visualise the train loss with opencv
		std::cout << "Train loss: " << trainLoss << std::endl;

		if (epoch % opts.valInterval == 0)
		{
			ret = lenet_val(model, valloader, bestValLoss, lossFn, valImprov, opts);
			if (ret == 1)
			{
				std::cout << "The training failed, fatal" << std::endl;
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
			std::cout << "Validation is unstable, change parameters" << std::endl;
			return 1;
		}
		i++;
		valLoss += loss.item<float>();
	}
	valLoss /= i;
	
	// TODO visualise
	std::cout << "The validation loss is: " << valLoss << std::endl;
	
	// < is used to avoid false improvements
	if (valLoss < bestValLoss)
	{
		// TODO: error handling if the path doesn't exist
		torch::save(model, mnistOpts.savepath);
		bestValLoss = valLoss;
		imp = true;
		std::cout << "The model improved" << std::endl;
	}

	return 0;
}

template<typename Dataloader>
int lenet_test(Dataloader &testloader, Settings &opts)
{
	std::cout << "Starting testing" << "\n";
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgresz);
	torch::load(model, mnistOpts.savepath);
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
		
	return 0;
}
