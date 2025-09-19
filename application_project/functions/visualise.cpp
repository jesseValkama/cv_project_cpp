#include "visualise.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <stdint.h>
#include <utility>

#include "../datasets/loader_funcs.h"
#include "../settings.h"

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, int64_t label, double prob, DatasetOpts datasetOpts, cv::ColormapTypes type)
{
	tfm.mul_(255);
	if (tfm.dim() == 2) { tfm.unsqueeze_(0); }
	std::optional<cv::Mat> fm = Tensor2mat(tfm, -1, false, std::make_pair(std::vector<double>{}, std::vector<double>{}));
	if (!fm.has_value()) { return 1; }
	std::optional<cv::Mat> inputImg = Tensor2mat(tInputImg, 0, false, std::make_pair(datasetOpts.mean, datasetOpts.stdev));
	if (!inputImg.has_value()) { return 1; }
	cv::ColorConversionCodes colourConv = (datasetOpts.numOfChannels == 3) ? cv::COLOR_BGR2RGB : cv::COLOR_GRAY2BGR;
	cv::cvtColor(*inputImg, *inputImg, colourConv);
	
	cv::Mat cm;
	cv::applyColorMap(*fm, cm, type);
	if ((*inputImg).size() != cm.size())
	{
		cv::resize(cm, cm, cv::Size((*inputImg).size()));
	}
	cv::addWeighted(*inputImg, 0.6, cm, 0.4, 0.0, cm);

	cv::resize(cm, cm, cv::Size(500, 500));

	std::string labelText = "Prediction: " + std::to_string(label) + " with the probability of: " + std::to_string(prob);
	cv::putText(cm, labelText, cv::Point(30, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,0,255), 2, cv::LINE_AA);
	cv::imshow("fmvis", cm);
	cv::waitKey(0);

	return 0;
}
