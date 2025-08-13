#include "train_lenet.h"

#include <iostream>
#include <memory>
#include <limits>
#include <utility>

#include <torch/torch.h>

#include "../datasets/mnist.h"
#include "../models/lenet.h"
#include "../settings.h"
#include "common.h"

namespace nn = torch::nn;

int lenet_loop(Settings &opts)
{
	LeNet model(opts.numOfChannels);
	model->to(mnistOpts.dev);
	
	// a custom dataset is a bit pointless, but it is used as "proof of concept" or if pose is ready, it is useful there
	// incase i am not done with this, i am overfitting to debug
	const auto data = readInfo(); // this needs to be improved
	auto dataset = MnistDataset(data.first).map(torch::data::transforms::Stack<>());
	auto trainSize = dataset.size().value();
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
		(
			std::move(dataset), 
			mnistOpts.trainBS
		);
	
	torch::optim::Adam optimiser(model->parameters(), torch::optim::AdamOptions(opts.learningRate));
	torch::nn::CrossEntropyLoss lossFn;
	
	int ret = lenet_train(model, *dataloader, *dataloader, optimiser, lossFn, opts);
	if (ret == 1)
	{
		return 1;
	}

	ret = lenet_test(model, *dataloader, lossFn);
	if (ret == 1)
	{
		std::cout << "The testing failed, fatal" << std::endl;
		return 1;
	}

	return 0;
}

template<typename Dataloader>
int lenet_train(LeNet &model, Dataloader &trainloader, Dataloader &valloader,
	torch::optim::Optimizer &optimiser, nn::CrossEntropyLoss &lossFn, Settings &opts)
{
	model->train();
	int ret = 0;
	float bestValLoss = std::numeric_limits<float>::infinity();
	bool valImprov = false;
	bool stop = false;
	size_t dec = 0;

	for (size_t epoch = 1; epoch <= opts.maxEpochs; epoch++)
	{	
		float trainLoss = 0.0;
		int i = 0;
		for (auto &batch : trainloader)
		{
			auto imgs = batch.data.to(mnistOpts.dev);
			auto labels = batch.target.to(mnistOpts.dev).view({-1});

			optimiser.zero_grad();

			torch::Tensor outputs = model->forward(imgs);
			auto loss = lossFn(outputs, labels);
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

		std::cout << "Epoch:" << epoch << std::endl;

		// TODO: visualise the train loss with opencv
		std::cout << "Train loss:" << trainLoss << std::endl;

		if (epoch % opts.valInterval == 0)
		{
			ret = lenet_val(model, valloader, bestValLoss, lossFn, valImprov);
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
int lenet_val(LeNet &model, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp)
{
	model->eval();
	int i = 0;
	float valLoss = 0.0;
	
	torch::NoGradGuard no_grad;
	for (auto& batch : valloader)
	{
		auto imgs = batch.data.to(mnistOpts.dev);
		auto labels = batch.target.to(mnistOpts.dev).view({-1});

		torch::Tensor outputs = model->forward(imgs);
		auto loss = lossFn(outputs, labels);
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
	std::cout << "The loss is:" << valLoss << std::endl;
	
	// < is used to avoid false improvements
	if (valLoss < bestValLoss)
	{
		torch::save(model, "../weights/tmp.pth");
		bestValLoss = valLoss;
		imp = true;
		std::cout << "The model improved" << std::endl;
	}

	return 0;
}

template<typename Dataloader>
int lenet_test(LeNet &model, Dataloader &testloader, nn::CrossEntropyLoss &lossFn)
{
	//todo make the entire functino
	model->eval();
	int i = 0;
	
	torch::NoGradGuard no_grad;
	for (auto &batch : testloader)
	{
		auto imgs = batch.data.to(mnistOpts.dev);
		auto labels = batch.target.to(mnistOpts.dev).view({-1});

		torch::Tensor outputs = model->forward(imgs);
		
		i++;
	}

	return 0;
}
