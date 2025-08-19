#include "mnist.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Batch MnistDataset::get(size_t i)
{
	/*
	* TODO:
	* scale between 0 and 1
	*/

	std::pair<cv::Mat, char> p = load_mnist_img(mnistOpts.fimgname, i, info, mnistOpts.imgsz, mnistOpts.imgsz);
	cv::resize(p.first, p.first, cv::Size(mnistOpts.imgresz, mnistOpts.imgresz));
	torch::Tensor timg = torch::from_blob
	(
		p.first.data,
		{ 1, mnistOpts.imgresz, mnistOpts.imgresz },
		torch::kUInt8
	).to(torch::kFloat);
	timg.div_(255);
	torch::Tensor tlabel = torch::tensor(p.second, torch::kLong);

	return {timg, tlabel};
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
