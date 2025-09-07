#ifndef LENET_INF
#define LENET_INF

#include <torch/torch.h>

#include <string>
#include <vector>

#include "../settings.h"

int run_inference(Settings &opts, int idx = -1);
/*
* Function to gather information about the images to be passed for inference
* Calls lenet_inference
* 
* Args:
*	opts: options for inference
*	idx: fm to visualise (-1 for gradcam, -2 to skip)
*
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts, int idx = -1);
/*
* Function for getting results of inference
* 
* Args:
*	fImgs: container for information about the images
*	opts: options for inference
*	idx: fm to visualise (-1 for gradcam, -2 to skip)
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/



#endif