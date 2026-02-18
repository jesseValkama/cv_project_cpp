#ifndef GRADCAM_H
#define GRADCAM_H

#include <torch/torch.h>

#include <optional>
#include <stdint.h>
#include <string>

#include "../settings.h"

int gradcam(torch::Tensor y, torch::Tensor &tfm, torch::Tensor &inputImg, int64_t label, double prob, DatasetOpts &datasetOpts, float imgWeight, bool save = true, std::string fname = "");
/*
* Student's implementation for gradcam
* Doesn't use hooks to make it simplier (not good for modularity)
* but would be optimal for research, because it is quick to implement
* see LeNet class for how it caches the feature maps and gradients
* 
* for more info about gradcam (official paper):
*	 https://arxiv.org/abs/1610.02391
* 
* Args:
*	y: the logit to be backpropagated on
*	tfm: feature map as a Tensor
*	inputImg: the input img as a Tensor
*	label: the label idx for the prediction
*	prob: the probability for the prediction
*	datasetOpts: settings for the dataset
*	save: whether to save the visualisations
*	fname: the name of the input img
* 
* Returns:
*	0: successfull
*	1: failed (logged to terminal)
*/

#endif