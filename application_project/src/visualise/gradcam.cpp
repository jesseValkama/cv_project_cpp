#include "visualise/gradcam.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <iostream>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>

#include "settings/settings.h"
#include "visualise/common.h"
#include "visualise/feature_map_visualiser.h"

int gradcam(torch::Tensor &imgs, torch::Tensor &featureMaps, std::unordered_map<std::string, torch::Tensor> &outputs,
	 DatasetOpts &datasetOpts, fmvis::FeatureMapVisualiserOptions &fmopts)
{
	auto batchSize = featureMaps.sizes()[0];
	torch::autograd::grad({outputs.at("logits").sum()}, {featureMaps}, {});
	torch::Tensor importanceWeights = featureMaps.grad().mean({ 2, 3 })
												 .unsqueeze_(-1).unsqueeze_(-1); // BCHW -> BC11
	torch::Tensor heatmaps = (importanceWeights * featureMaps).sum(1, true); // BCHW * BC11 -> B1HW
	heatmaps = torch::relu(heatmaps);
	return visualise_fm(imgs, heatmaps, outputs, datasetOpts, fmopts);
}