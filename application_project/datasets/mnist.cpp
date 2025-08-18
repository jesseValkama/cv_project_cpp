#include "mnist.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Example MnistDataset::get(size_t i)
{
	/*
	* TODO:
	* add normalisation (z-scaling)
	* add shuffle
	*/

	std::pair<cv::Mat, char> p = load_mnist_img(mnistOpts.fimgname, i, info, mnistOpts.imgsz, mnistOpts.imgsz);
	cv::resize(p.first, p.first, cv::Size(mnistOpts.imgresz, mnistOpts.imgresz));
	auto timg = torch::from_blob
	(
		p.first.ptr(),
		{ 1, mnistOpts.imgresz, mnistOpts.imgresz },
		torch::kUInt8
	).to(torch::kFloat);
	auto tlabel = torch::tensor(p.second, torch::kLong);

	return {timg, tlabel};
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
