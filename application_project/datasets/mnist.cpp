#include "mnist.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "loader_funcs.h"

#include "../settings.h"

Example MnistDataset::get(size_t i)
{
	std::pair<cv::Mat, char> p = load_mnist_img(mnistOpts.fimgname, i, info, 28, 28);
	
	// works this far, but this code is for RGB -> finally time to fix this
	cv::resize(p.first, p.first, cv::Size(mnistOpts.imgsz, mnistOpts.imgsz));
	std::vector<cv::Mat> channels(3);
	cv::split(p.first, channels);

	auto B = torch::from_blob
	(
		channels[0].ptr(),
		{ mnistOpts.imgsz, mnistOpts.imgsz },
		torch::kUInt8
	);
	auto G = torch::from_blob
	(
		channels[1].ptr(),
		{ mnistOpts.imgsz, mnistOpts.imgsz },
		torch::kUInt8
	);
	auto R = torch::from_blob
	(
		channels[2].ptr(),
		{ mnistOpts.imgsz, mnistOpts.imgsz },
		torch::kUInt8
	);

	auto timg = torch::cat({ R, G, B })
		.view({ 3, mnistOpts.imgsz, mnistOpts.imgsz })
		.to(torch::kFloat);
	auto tlabel = torch::tensor(p.second, torch::kLong);

	return {timg, tlabel};
}

torch::optional<size_t> MnistDataset::size() const
{
	return info.size();
}
