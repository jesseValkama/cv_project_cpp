#include "visualise.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "../datasets/loader_funcs.h"

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, cv::ColormapTypes type)
{
	std::optional<cv::Mat> fm = Tensor2greyscale(tfm, false, std::make_pair(-1.0, -1.0)); // double check this
	if (!fm.has_value()) { return 1; }
	std::optional<cv::Mat> inputImg = Tensor2greyscale(tInputImg, true);
	if (!inputImg.has_value()) { return 1; }
	cv::cvtColor(inputImg.value(), inputImg.value(), cv::COLOR_GRAY2BGR);
	
	cv::Mat cm;
	cv::applyColorMap(fm.value(), cm, type);
	if (inputImg.value().size() != cm.size())
	{
		cv::resize(cm, cm, cv::Size(inputImg.value().size()));
	}
	cv::addWeighted(inputImg.value(), 0.4, cm, 0.6, 0.0, cm);

	cv::resize(cm, cm, cv::Size(500, 500));
	cv::imshow("fmvis", cm);
	cv::waitKey(0);

	return 0;
}
