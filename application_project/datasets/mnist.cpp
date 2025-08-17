#include "mnist.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Example MnistDataset::get(size_t i)
{
	std::pair<cv::Mat, char> p = load_mnist_img(mnistOpts.fimgname, i, info, mnistOpts.imgsz, mnistOpts.imgsz);
	cv::resize(p.first, p.first, cv::Size(mnistOpts.imgsz, mnistOpts.imgsz));
	auto timg = torch::from_blob
	(
		p.first.ptr(),
		{ 1, mnistOpts.imgsz, mnistOpts.imgsz },
		torch::kUInt8
	).to(torch::kFloat);
	auto tlabel = torch::tensor(p.second, torch::kLong);

	return {timg, tlabel};
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
