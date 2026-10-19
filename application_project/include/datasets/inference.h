#ifndef INFERENCEDATASET_H
#define INFERENCEDATASET_H

#include <torch/torch.h>

#include <vector>

#include "loader_funcs.h"
#include "settings/settings.h"

class InferenceDataset : public torch::data::datasets::Dataset<InferenceDataset>
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
private:
	std::vector<std::string> imgPaths;
	DatasetOpts dsOpts;
	bool async = false;
public:
	InferenceDataset(const std::vector<std::string> &imgPaths, const DatasetOpts &dsOpts, const bool async) 
		: imgPaths(imgPaths), dsOpts(dsOpts), async(async) {}
			
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

#endif // INFERENCEDATASET_H