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

	// TODO: move to train fn, and load in test to allow for retesting experiments
	LeNet model(opts.numOfChannels, mnistOpts.imgresz);
	model->to(mnistOpts.dev);
	
	// a custom dataset is a bit pointless, but it is used as "proof of concept" or if pose is ready, it is useful there
	// incase i am not done with this, i am overfitting to debug
	Info trainInfo, testInfo;
	int status = 0;
	status = load_mnist_info(opts.mnistOpts, trainInfo, "train");
	if (status != 0)
	{
		return status;
	}

	// source for the hardcoded mean and stdev values: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python 
	auto dataset = MnistDataset(trainInfo, mnistOpts)
		.map(torch::data::transforms::Normalize<>(
			{0.1307}, {0.3081}))
		.map(torch::data::transforms::Stack<>());
	const size_t TRAIN_SIZE = dataset.size().value();
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
		(
			std::move(dataset),
			torch::data::DataLoaderOptions().batch_size(mnistOpts.trainBS).workers(mnistOpts.numWorkers)
		);
	
	torch::optim::Adam optimiser(model->parameters(), torch::optim::AdamOptions(opts.learningRate));
	torch::nn::CrossEntropyLoss lossFn;
	
	int ret = lenet_train(model, *dataloader, *dataloader, TRAIN_SIZE, optimiser, lossFn, opts);
	if (ret == 1)
	{
		return 1;
	}

	ret = lenet_test(model, *dataloader, lossFn, opts);
	if (ret == 1)
	{
		std::cout << "The testing failed, fatal" << std::endl;
		return 1;
	}

	return 0;
}

template<typename Dataloader>
int lenet_train(LeNet& model, Dataloader& trainloader, Dataloader& valloader, const size_t TRAIN_SIZE,
	torch::optim::Optimizer &optimiser, nn::CrossEntropyLoss &lossFn, Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
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
			torch::Tensor imgs = batch.data.to(mnistOpts.dev);
			torch::Tensor labels = batch.target.to(mnistOpts.dev).view({ -1 });
			
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
			ret = lenet_val(model, valloader, TRAIN_SIZE, bestValLoss, lossFn, valImprov, opts);
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
int lenet_val(LeNet &model, Dataloader &valloader, const size_t TRAIN_SIZE, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	model->eval();
	int i = 0;
	float valLoss = 0.0;
	
	torch::NoGradGuard no_grad;
	for (Batch &batch : valloader)
	{
		torch::Tensor imgs = batch.data.to(mnistOpts.dev);
		torch::Tensor labels = batch.target.to(mnistOpts.dev).view({-1});

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
int lenet_test(LeNet &model, Dataloader &testloader, nn::CrossEntropyLoss &lossFn, Settings &opts)
{
	std::cout << "Starting testing" << "\n";
	MnistOpts mnistOpts = opts.mnistOpts;
	model->eval();
	uint32_t nc = 0;

	std::unordered_map<std::string, uint32_t> cmstats =
	{ {"tp", 0}, {"fp", 0}, {"fn", 0} };

	torch::NoGradGuard no_grad;
	for (Batch &batch : testloader)
	{
		torch::Tensor imgs = batch.data.to(mnistOpts.dev);
		torch::Tensor labels = batch.target.to(mnistOpts.dev).view({-1});

		torch::Tensor outputs = model->forward(imgs);
		std::vector<std::unordered_map<std::string, uint32_t>> tmp = calc_cm(labels, outputs);
		nc = tmp.size();

		for (int i = 0; i < nc; ++i)
		{
			cmstats["tp"] += tmp[i]["tp"];
			cmstats["fp"] += tmp[i]["fp"];
			cmstats["fn"] += tmp[i]["fn"];
		}
	}

	std::cout << "tp: " << cmstats["tp"] << std::endl;
	std::cout << "fp: " << cmstats["fp"] << std::endl;
	std::cout << "fn: " << cmstats["fn"] << std::endl;
	
	return 0;
}
