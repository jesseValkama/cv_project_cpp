#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include <torch/torch.h>

#include "loader_funcs.h"

class MnistDataset : public torch::data::datasets::Dataset<MnistDataset>
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
	const DatasetOpts mnistOpts;
	const std::string type;
	const bool async = false;

	public:
		MnistDataset(const Info& info, const DatasetOpts& mnistOpts, const std::string type, const bool async) 
			: info(info), mnistOpts(mnistOpts), type(type), async(async) {}
				
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