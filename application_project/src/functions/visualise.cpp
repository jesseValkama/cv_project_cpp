#include "functions/visualise.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <cassert>
#include <stdint.h>
#include <string>
#include <utility>

#include "datasets/loader_funcs.h"
#include "settings.h"

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, int64_t label, double prob, DatasetOpts datasetOpts, cv::ColormapTypes type, float imgWeight, bool save, std::string fname)
{
	assert(imgWeight >= 0 && imgWeight <= 1);
	tfm.mul_(255);
	if (tfm.dim() == 2) { tfm.unsqueeze_(0); }
	std::optional<cv::Mat> fm = Tensor2mat(tfm, -1, false, std::make_pair(std::vector<double>{}, std::vector<double>{}));
	if (!fm.has_value()) { return 1; }
	std::optional<cv::Mat> inputImg = Tensor2mat(tInputImg, 0, false, std::make_pair(datasetOpts.mean, datasetOpts.stdev), 255.0f); // cannot normalise here because type is uchar
	if (!inputImg.has_value()) { return 1; }
	cv::ColorConversionCodes colourConv = (datasetOpts.numOfChannels == 3) ? cv::COLOR_RGB2BGR : cv::COLOR_GRAY2BGR;
	cv::cvtColor(*inputImg, *inputImg, colourConv);
	
	cv::Mat cm;
	cv::applyColorMap(*fm, cm, type);
	if (inputImg->size() != cm.size())
	{
		cv::resize(cm, cm, cv::Size((*inputImg).size()));
	}
	
	// calc is inspired by https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file
	cm.convertTo(cm, CV_32F);
	inputImg->convertTo(*inputImg, CV_32F);
	cm = cm / 255.0f;
	*inputImg = *inputImg / 255.0f;

	double max = 0.0;
	cv::minMaxLoc(inputImg->reshape(1), NULL, &max, NULL, NULL);
	assert(max <= 1.0);

	cm = (1.0f - imgWeight) * cm + imgWeight * (*inputImg);
	cv::minMaxLoc(cm.reshape(1), NULL, &max, NULL, NULL);
	cm = cm / static_cast<float>(max) * 255.0f;
	cm.convertTo(cm, CV_8U);
	int fProb = static_cast<int>(prob * 100);

	cv::resize(cm, cm, cv::Size(500, 500));
	std::string labelText = "Prediction: " + datasetOpts.labels[label] + " with the probability of: " + std::to_string(fProb) + "%";
	cv::putText(cm, labelText, cv::Point(30, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,0,255), 2, cv::LINE_AA);

	if (save)
	{
		if (fname == "") 
		{
			std::cout << "Invalid savepath" << std::endl;
			return 1;
		}
		cv::imwrite("D:/datasets/saved_visualisations/" + fname, cm); // TODO: do not hardcode
	}
	cv::imshow("fmvis", cm);
	cv::waitKey(0);

	return 0;
}
