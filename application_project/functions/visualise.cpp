#include "visualise.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <stdint.h>

#include "../datasets/loader_funcs.h"

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, int64_t label, double prob, cv::ColormapTypes type)
{
	if (tfm.dim() == 2) { tfm.unsqueeze_(0); }
	std::optional<cv::Mat> fm = Tensor2mat(tfm, -1, std::make_pair(std::vector<double>{}, std::vector<double>{}));
	if (!fm.has_value()) { return 1; }
	std::optional<cv::Mat> inputImg = Tensor2mat(tInputImg, 0);
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

	std::string labelText = "Prediction: " + std::to_string(label) + " with the probability of: " + std::to_string(prob);
	cv::putText(cm, labelText, cv::Point(30, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,0,255), 2, cv::LINE_AA);
	cv::imshow("fmvis", cm);
	cv::waitKey(0);

	return 0;
}
