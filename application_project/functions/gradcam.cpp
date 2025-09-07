#include "gradcam.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <iostream>
#include <optional>

#include "visualise.h"

int gradcam(torch::Tensor y, torch::Tensor &fmAct, torch::Tensor &inputImg)
{
	y.backward({}, true); // unoptimised, backpropagates the entire model, not til the hook
	torch::Tensor alpha = fmAct.grad().mean({ 2, 3 });

	alpha.squeeze_(0).unsqueeze_(1).unsqueeze_(1);
	fmAct.squeeze_(0);
	
	torch::Tensor localMaps = (alpha * fmAct).sum(0);
	localMaps = torch::relu(localMaps);

	int ret = visualise_fm(localMaps, inputImg, cv::COLORMAP_JET);
	if (ret != 0) { return ret; }
	
	return 0;
}