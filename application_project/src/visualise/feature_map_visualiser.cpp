#include "visualise/feature_map_visualiser.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <cassert>
#include <stdint.h>
#include <string>
#include <optional>
#include <unordered_map>
#include <utility>

#include "datasets/loader_funcs.h"
#include "settings/settings.h"
#include "visualise/common.h"
#include "visualise/gradcam.h"

namespace idx = torch::indexing;

int fmvis::FeatureMapVisualiser::visualise(torch::Tensor &imgs, torch::Tensor &featureMaps, std::unordered_map<std::string, torch::Tensor> &outputs)
{
    int ret = 0;
    for (fmvis::Visualisers task : tasks)
    {
		this->fmopts.task = task;
        switch (task.task)
        {
            case IDX:
            {
				featureMaps = featureMaps.index({idx::Slice(), idx::Slice(*task.idx, *task.idx + 1), idx::Slice(), idx::Slice()});
                ret = visualise_fm(imgs, featureMaps, outputs, this->dsOpts, this->fmopts);
                if (ret != 0)
                {
                    return ret;
                }
				break;
            }
            case GRADCAM:
            {
                ret = gradcam(imgs, featureMaps, outputs, this->dsOpts, this->fmopts);
                if (ret != 0)
                {
                    return ret;
                }
				break;
            }
			default:
			{
				return 1;
			}
        }
    }
	return 1;
}


int visualise_fm(torch::Tensor &imgs, torch::Tensor &featureMaps, std::unordered_map<std::string, torch::Tensor> &outputs,
	DatasetOpts &dsOpts, fmvis::FeatureMapVisualiserOptions &fmopts)
{
	assert(fmopts.imgWeight >= 0 && fmopts.imgWeight <= 1);
	featureMaps.mul_(255);
	if (featureMaps.dim() == 2) { featureMaps.unsqueeze_(0); }

	auto n = featureMaps.sizes()[0];
	for (int i = 0; i < n; ++i)
	{
		std::optional<cv::Mat> fm = Tensor2mat(featureMaps[i], std::nullopt, false, std::make_pair(std::vector<double>{}, std::vector<double>{}));
		if (!fm.has_value()) { return 1; }
		std::optional<cv::Mat> img = Tensor2mat(imgs[i], std::nullopt, false, std::make_pair(dsOpts.mean, dsOpts.stdev), 255.0f);
		if (!img.has_value()) { return 1; }
		cv::ColorConversionCodes colourConv = (dsOpts.numOfChannels == 3) ? cv::COLOR_RGB2BGR : cv::COLOR_GRAY2BGR;
		cv::cvtColor(*img, *img, colourConv);
		
		cv::Mat cm;
		cv::applyColorMap(*fm, cm, fmopts.task->cmap);
		if (img->size() != cm.size())
		{
			cv::resize(cm, cm, cv::Size((*img).size()));
		}
		
		// calc is inspired by https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file
		cm.convertTo(cm, CV_32F);
		img->convertTo(*img, CV_32F);
		cm = cm / 255.0f;
		*img = *img / 255.0f;

		double max = 0.0;
		cv::minMaxLoc(img->reshape(1), NULL, &max, NULL, NULL);
		assert(max <= 1.0);

		cm = (1.0f - fmopts.imgWeight) * cm + fmopts.imgWeight * (*img);
		cv::minMaxLoc(cm.reshape(1), NULL, &max, NULL, NULL);
		cm = cm / static_cast<float>(max) * 255.0f;
		cm.convertTo(cm, CV_8U);
		double prob = outputs.at("probs")[i].item<double>();
		int idx = outputs.at("idxs")[i].item<int>();

		cv::resize(cm, cm, cv::Size(500, 500));
		std::string labelText = "Prediction: " + dsOpts.labels[idx] + " with the probability of: " + std::to_string(prob) + "%";
		cv::putText(cm, labelText, cv::Point(30, 30), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,0,255), 2, cv::LINE_AA);

		if (fmopts.savepath.has_value())
		{
			cv::imwrite(*fmopts.savepath, cm);
		}
		cv::imshow("fmvis", cm);
		cv::waitKey(0);
	}
	return 0;
}
