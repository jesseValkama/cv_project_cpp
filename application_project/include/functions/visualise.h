#ifndef VISUALISE_H
#define VISUALISE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <stdint.h>
#include <string>
#include "../settings.h"

int visualise_fm(torch::Tensor &tfm, torch::Tensor &tInputImg, int64_t label, double prob, DatasetOpts datasetOpts, cv::ColormapTypes type = cv::COLORMAP_JET, float imgWeight = 0.5, bool save = true, std::string fname = "");
/*
* Used to visualise a feature map with opencv
* Requires an existing window with the name of fmvis (which it doesn't destroy)
* 
* Credit:
*	some of the denormalisations are inspired by the show_cam_on_image function from the pytroch-grad-cam package
*	https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file
* 
* Args:
*	tfm: feature map as a tensor
*	tInputImg: the inputimg as a tensor
*	label: the idx of the label for the prediction
*	prob: the probability for the prediction
*	datasetOpts: settings for the dataset
*	type: which colormap to use, see https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
*	imgWeight: the percentage for the input image (colour map is 1 - imgWeight)
*	save: whether to save the visualisation
*	fname: the name of the input img, can't be left default if save
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
* 
* Saves:
*	the feature map visualisation if save
*/

#endif