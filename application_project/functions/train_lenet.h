#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <torch/torch.h>

#include <memory>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../models/lenet.h"
#include "../models/model_wrapper.h"
#include "../settings.h"

int lenet_loop(Settings &opts, bool train, bool test);
/*
	* This is the main training loop
	* 
	* Args:
	*	opts: options for training
	*	train: bool to train a model
	*	test: bool to test a model
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/

template<typename Dataloader>
int lenet_train(Dataloader &trainloader, Dataloader &valloader, Settings &opts);
/*
	* The function to train LeNet
	* 
	* Args:
	*	trainloader: then dataloader for training
	*	valloader: the dataloader for validation (passed to lenet_val)
	*	opts: options for training
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/

template<typename Dataloader>
int lenet_val(std::shared_ptr<ModelWrapper> model, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts);
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

template<typename Dataloader>
int lenet_test(Dataloader &testloader, Settings &opts);
/*
	* The function to test LeNet, currently only tests the model from the last training
	* 
	* Args:
	*	testloader: dataloader for testing
	*	opts: settings for testing
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/

#endif