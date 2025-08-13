#ifndef MNIST_H
#define MNIST_H

#include <torch/torch.h>
#include <stdint.h>

#include "../settings.h"

typedef std::vector<std::pair<std::string, int64_t>> Data;
typedef torch::data::Example<> Example;

static MnistOptions mnistOpts;

class MnistDataset : public torch::data::datasets::Dataset<MnistDataset>
{
	const Data data;

	public:
		MnistDataset(const Data& data) : data(data) {}
		Example get(size_t i);
		torch::optional<size_t> size() const;
};

std::pair<Data, Data> readInfo(void);

#endif