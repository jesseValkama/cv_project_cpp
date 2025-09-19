#include "cifar10.h"

#include <cassert>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "../augmentations/geometry.h"
#include "../settings.h"
#include "loader_funcs.h"

Batch Cifar10Dataset::get(size_t i)
{
	assert(type == "train" || type == "val" || type == "test");
	
	uint8_t batchIdx = 255;
	for (uint8_t j = 0; j < idxs.size(); ++j) // runtime doens't matter due to low batch n
	{
		if (i < idxs[j])
		{
			batchIdx = j;
			break;
		}
	}
	const std::string path = (type == "train" || type == "val") ? cifar10Opts.fTrain[batchIdx] : cifar10Opts.fTest[batchIdx]; // train and val are in the train file, they are split
	const int imgresz = (cifar10Opts.imgsz == cifar10Opts.imgresz) ? -1 : cifar10Opts.imgresz;

	std::optional<std::pair<cv::Mat, char>> p = load_mnist_img(path, i, info, cifar10Opts.imgsz, cifar10Opts.imgsz, imgresz, cifar10Opts.numOfChannels, true); // unoptmised fn, need to have a look
	if (p.has_value())
	{
		aug::mirror((*p).first, 1);
		aug::mirror((*p).first, 0, 0.2);
		aug::crop((*p).first);
		torch::Tensor timg = mat2Tensor((*p).first, cifar10Opts.imgresz, cifar10Opts.numOfChannels, 255, true);
		torch::Tensor tlabel = torch::tensor((*p).second, torch::kLong);
		if (async)
		{
			timg = timg.pin_memory();
			tlabel = tlabel.pin_memory();
		}
		return {timg, tlabel};
	}
	std::abort();
}

torch::optional<size_t> Cifar10Dataset::size() const
{
	return info.size();
}
