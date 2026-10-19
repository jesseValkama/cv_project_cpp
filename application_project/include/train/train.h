#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <cstddef>

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "datasets/loader_funcs.h"
#include "models/model_wrapper.h"
#include "optims/scheds/warmup.h"
#include "settings/settings.h"
#include "train/common.h"
#include "train/early_stop.h"
#include "train/grad_scaler.h"

/*
* Inspiration:
*	https://github.com/pytorch/examples/tree/main/cpp/mnist
*/

namespace nn = torch::nn;

template<typename Randomloader, typename Sequentialloader>
int train_loop(Randomloader &trainloader, Sequentialloader &valloader, train::Tracker &tracker, const size_t trainSize, 
	Settings &opts, ModelTypes modelType)
/*
	* The function to train LeNet
	* 
	* Args:
	*	trainloader: then dataloader for training
	*	valloader: the dataloader for validation (passed to lenet_val)
	*	trainSize: the size of the trainset
	*	opts: options for training]
	*	modelType: the type of the model
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::shared_ptr<ModelWrapper> modelWrapper = std::make_shared<ModelWrapper>(modelType, mnistOpts);
	modelWrapper->to(opts.dev);
	torch::optim::AdamW optimiser(modelWrapper->parameters(), torch::optim::AdamWOptions(opts.learningRate).weight_decay(opts.weightDecay));
	torch::optim::ReduceLROnPlateauScheduler plateau(optimiser, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min, 0.1, opts.plateauWait);
	optims::sched::warmupLR warmup(optimiser, opts.learningRate, opts.warmupLen * trainSize / mnistOpts.trainBS);
	nn::CrossEntropyLoss lossFn;
	amp::GradScaler scaler = amp::GradScaler(opts.dev, 65536.0, 2.0, 0.5, 2000, opts.gradScaler);

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
			
			at::autocast::set_autocast_enabled(opts.devType, opts.amp);
			at::autocast::set_autocast_dtype(opts.devType, opts.ampDtype);
			torch::Tensor outputs = modelWrapper->forward(imgs);
			torch::Tensor loss = lossFn(outputs, labels);
			if (std::isnan(loss.template item<double>()))
			{
				std::cout << ANSI_RED << "Training is unstable, change settings" << ANSI_END << std::endl;
				return 1;
			}
			at::autocast::set_autocast_enabled(opts.devType, false);
			scaler.scale(loss).backward();
			scaler.step(optimiser);
			scaler.update();

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

#endif