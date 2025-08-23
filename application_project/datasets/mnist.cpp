#include "mnist.h"

#include <cassert>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Batch MnistDataset::get(size_t i)
{
	assert(type == "train" || type == "val" || type == "test");
	
	// train and val in the same file
	std::string fImgs = (type == "train" || type == "val") ? mnistOpts.fTrainImgs : mnistOpts.fTestImgs;

	std::pair<cv::Mat, char> p = load_mnist_img(fImgs, i, info, mnistOpts.imgsz, mnistOpts.imgsz, mnistOpts.imgresz);
	torch::Tensor timg = greyscale2Tensor(p.first, mnistOpts.imgresz, 255);
	torch::Tensor tlabel = torch::tensor(p.second, torch::kLong);

	return {timg, tlabel};
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
