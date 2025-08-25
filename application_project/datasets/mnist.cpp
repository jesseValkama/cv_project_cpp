#include "mnist.h"

#include <cassert>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Batch MnistDataset::get(size_t i)
{
	assert(type == "train" || type == "val" || type == "test");
	
	// train and val in the same file
	std::string fImgs = (type == "train" || type == "val") ? mnistOpts.fTrainImgs : mnistOpts.fTestImgs;
	
	// console notifies if problems arise with opening stream or img
	int n = info.size();
	for (size_t j = i; j < n; ++j)
	{
		std::optional<std::pair<cv::Mat, char>> p = load_mnist_img(fImgs, j, info, mnistOpts.imgsz, mnistOpts.imgsz, mnistOpts.imgresz);
		if (p.has_value())
		{
			torch::Tensor timg = greyscale2Tensor(p.value().first, mnistOpts.imgresz, 255);
			torch::Tensor tlabel = torch::tensor(p.value().second, torch::kLong);
			return {timg, tlabel};
		}
	}
	// crashes if data is corrupt
	std::abort();
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
