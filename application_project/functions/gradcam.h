#ifndef GRADCAM_H
#define GRADCAM_H

#include <torch/torch.h>

#include <optional>

int gradcam(torch::Tensor y, torch::Tensor &tfm, torch::Tensor &inputImg);
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
* 
* Returns:
*	0: successfull
*	1: failed (logged to terminal)
*/

#endif