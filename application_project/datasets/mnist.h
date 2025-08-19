#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include <torch/torch.h>

#include "loader_funcs.h"

class MnistDataset : public torch::data::datasets::Dataset<MnistDataset>
{
	/*
		* Code for the MNIST dataset with imgs of 28 x 28
		* Args:
		* info: contains a vector of pairs with each imgs relative
		* position within the ubyte files and a corresponding label
		* 
		* mnistOpts: constains info about training like batchsizes and imgsizes
		* 
	*/
	const Info info;
	const MnistOpts mnistOpts;

	public:
		MnistDataset(const Info& info, const MnistOpts& mnistOpts) 
			: info(info), mnistOpts(mnistOpts) {}
		
		Batch get(size_t i);
		torch::optional<size_t> size() const;
};

#endif