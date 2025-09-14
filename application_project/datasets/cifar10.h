#ifndef CIFAR10DATASET_H
#define CIFAR10DATASET_H

#include <torch/torch.h>

#include "loader_funcs.h"
#include "../settings.h"

class Cifar10Dataset : public torch::data::datasets::Dataset<Cifar10Dataset>
{
	/*
	* Custom dataset for mnist
	* 
	* Attributes:
	*	info: container for the img and label positions in the ubyte file
	*	minstOpts: options for loading the images such as sizes
	*	type: determines which file to open, train or test (train and val) are
	*	read from the same file
	* 
	* Inspiration:
	*	https://github.com/pytorch/examples/tree/main/cpp/custom-dataset 
	*/

	const Info info;
	const DatasetOpts cifar10Opts;
	const std::string type;

	public:
		Cifar10Dataset(const Info &info, const DatasetOpts &cifar10Opts, const std::string type)
			: info(info), cifar10Opts(cifar10Opts), type(type) {}
				
		Batch get(size_t i);
		/*
		* Get method used by the libtorch dataloader, by loading imgs from dataset info
		* Corrupt images will be skipped and logged to the terminal
		* 
		* Args:
		*	i: the index of the img and label in the container
		* 
		* Returns:
		*	batch: img as a tensor and the corresponding label (only a single pair)
		*/
		torch::optional<size_t> size() const;
		/*
		* Gets the length of the container (n of imgs and labels)
		* 
		* Returns:
		*	the length of the container
		*/
};

#endif