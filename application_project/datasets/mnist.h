#ifndef MNISTDATASET_H
#define MNISTDATASET_H

#include <torch/torch.h>

#include "loader_funcs.h"

class MnistDataset : public torch::data::datasets::Dataset<MnistDataset>
{
	const Info info;
	const MnistOpts mnistOpts;

	public:
		MnistDataset(const Info& info, const MnistOpts& mnistOpts) 
			: info(info), mnistOpts(mnistOpts) {}
		Example get(size_t i);
		torch::optional<size_t> size() const;
};

#endif