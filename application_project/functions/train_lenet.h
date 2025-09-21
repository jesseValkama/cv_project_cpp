#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <torch/torch.h>

#include <memory>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../datasets/data_helper.h"
#include "../models/lenet.h"
#include "../models/model_wrapper.h"
#include "../settings.h"

/*
* Inspiration:
*	https://github.com/pytorch/examples/tree/main/cpp/mnist
*/

int lenet_loop(Settings &opts, ModelTypes modelType, DatasetTypes datasetType, bool train = false, bool test = false);
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

int mnist_loop(Settings &opts, ModelTypes modelType, const Info &trainInfo, const Info &valInfo, const Info &testInfo, bool train = false, bool test = false);

int cifar10_loop(Settings &opts, ModelTypes modelType, const Info &trainInfo, const Info &valInfo, const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs, bool train = false, bool test = false);

template<typename Randomloader, typename Sequentialloader>
int lenet_train(Randomloader &trainloader, Sequentialloader &valloader, Settings &opts, ModelTypes modelType);
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

template<typename Sequentialloader>
int lenet_val(std::shared_ptr<ModelWrapper> model, Sequentialloader &valloader, double &bestValLoss, double &valLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts);
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

template<typename Sequentialloader>
int lenet_test(Sequentialloader &testloader, Settings &opts, ModelTypes modelType, bool train);
/*
	* The function to test LeNet, currently only tests the model from the last training
	* 
	* Args:
	*	testloader: dataloader for testing
	*	opts: settings for testing
	*	modelType: which model
	*	train: which model to test (trained or defined)
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/

#endif