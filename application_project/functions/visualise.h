#ifndef VISUALISE_H
#define VISUALISE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, cv::ColormapTypes type = cv::COLORMAP_JET);
/*
* Used to visualise a feature map with opencv
* Requires an existing window with the name of fmvis (which it doesn't destroy)
* 
* Args:
*	tfm: feature map as a tensor
*	iInputImg: the inputimg as a tensor
*	type: which colormap to use, see https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

#endif