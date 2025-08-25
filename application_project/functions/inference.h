#ifndef LENET_INF
#define LENET_INF

#include <torch/torch.h>

#include <string>
#include <vector>

#include "../settings.h"

int run_inference(Settings &opts);
/*
* Function to gather information about the images to be passed for inference
* Calls lenet_inference
* 
* Args:
*	opts: options for inference
*
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts);
/*
* Function for getting results of inference
* 
* Args:
*	fImgs: container for information about the images
*	opts: options for inference
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

void visualise_fm(torch::Tensor tfm);
/*
* Visualises the feature map with openVC
* 
* Args:
*	tfm: feature map as a tensor
*/

#endif