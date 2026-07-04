#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <torch/torch.h>

#include <limits>
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

/*
* TODO: check const usage
*/

namespace train
{

struct Tracker
{
	double trainLoss = std::numeric_limits<double>::infinity();
	double valLoss = std::numeric_limits<double>::infinity();
	double bestValLoss = std::numeric_limits<double>::infinity();
	double accuracy = 0.0;
	size_t epoch;
	float trainTime;
	float testTime;
};

int run_loop(Settings &opts, const ModelTypes modelType, const DatasetTypes datasetType, 
	const bool trainModel = false, const bool testModel = false);
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

} // end of train

int mnist_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, train::Tracker &tracker, const bool trainModel = false, const bool testModel = false);
/*
*/

int cifar10_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs, 
	train::Tracker &tracker, const bool trainModel = false, const bool testModel = false);
/*
*/

template<typename Randomloader, typename Sequentialloader>
int train_loop(Randomloader &trainloader, Sequentialloader &valloader, train::Tracker &tracker, const size_t trainSize, 
	Settings &opts, ModelTypes modelType);
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

template<typename Sequentialloader>
int val_loop(std::shared_ptr<ModelWrapper> model, Sequentialloader &valloader, train::Tracker &tracker, 
	nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts);
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
int test_loop(Sequentialloader &testloader, Settings &opts, const ModelTypes modelType, const std::string &datasetName,
	 train::Tracker &tracker, const bool useTrainedModel);
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