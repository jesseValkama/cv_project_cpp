#ifndef RUNLOOP_H
#define RUNLOOP_H

#include <vector>

#include "datasets/loader_funcs.h"
#include "datasets/data_helper.h"
#include "settings/settings.h"
#include "train/common.h"

namespace train
{

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

// start of tmp

int mnist_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, train::Tracker &tracker, const bool trainModel = false, const bool testModel = false);
/*
*/

int cifar10_helper(Settings &opts, const ModelTypes modelType, const Info &trainInfo, const Info &valInfo, 
	const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs, 
	train::Tracker &tracker, const bool trainModel = false, const bool testModel = false);
/*
*/

// end of tmp

#endif // RUNLOOP_H