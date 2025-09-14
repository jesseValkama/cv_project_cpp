#include "cifar10.h"

#include <cassert>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Batch Cifar10Dataset::get(size_t i)
{
	assert(type == "train" || type == "val" || type == "test");
	
	uint8_t batchIdx = static_cast<uint8_t>(i / 1000);
	std::string path = (type == "train" || type == "val") ? cifar10Opts.fTrain[batchIdx] : cifar10Opts.fTest[batchIdx]; // train and val are in the train file, they are later split
	int imgresz = (cifar10Opts.imgsz == cifar10Opts.imgresz) ? -1 : cifar10Opts.imgresz;
	std::optional<std::pair<cv::Mat, char>> p = load_mnist_img(path, i, info, cifar10Opts.imgsz, cifar10Opts.imgsz, imgresz, cifar10Opts.numOfChannels); // unoptmised fn, need to have a look
	if (p.has_value())
	{
		torch::Tensor timg = mat2Tensor(p.value().first, cifar10Opts.imgresz, cifar10Opts.numOfChannels, 255).pin_memory();
		torch::Tensor tlabel = torch::tensor(p.value().second, torch::kLong).pin_memory();
		return {timg, tlabel};
	}
	std::abort();
}

torch::optional<size_t> Cifar10Dataset::size() const
{
	return info.size();
}
