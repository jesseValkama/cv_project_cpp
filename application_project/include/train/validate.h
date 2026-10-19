#ifndef VALLOOP_H
#define VALLOOP_H

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "datasets/loader_funcs.h"
#include "models/model_wrapper.h"
#include "settings/settings.h"
#include "train/common.h"

template<typename Sequentialloader>
int val_loop(std::shared_ptr<ModelWrapper> modelWrapper, Sequentialloader &valloader, train::Tracker &tracker, 
	nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts)
/*
	* The function to validate LeNet
	* 
	* Args:
	*	model: the model to be validated (created in train)
	*	valloader: dataloader for validation
	*	bestValLoss: the best value to track improvements
	*	lossFn: fn used to calculate loss
	*	imp: bool to track improvement from current validation
	*	opts: options for validation
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to teminal)
*/
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	modelWrapper->eval();
	int i = 0;
	
	torch::NoGradGuard no_grad;
	for (Batch &batch : valloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, mnistOpts.async);
		torch::Tensor labels = batch.target.to(opts.dev, mnistOpts.async).view({-1});

		at::autocast::set_autocast_enabled(opts.devType, opts.amp);
		at::autocast::set_autocast_dtype(opts.devType, torch::kFloat16);
		torch::Tensor outputs = modelWrapper->forward(imgs);
		torch::Tensor loss = lossFn(outputs, labels);
		if (std::isnan(loss.template item<double>()))
		{
			std::cout << ANSI_RED << "Validation is unstable, change parameters" << ANSI_END << std::endl;
			return 1;
		}
		i++;
		tracker.valLoss += loss.item<double>();
		at::autocast::set_autocast_enabled(opts.devType, false);
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

#endif // VALLOOP_H