#ifndef LENET_INF
#define LENET_INF

#include <torch/torch.h>

#include <string>
#include <vector>

#include "../settings.h"
#include "../models/model_wrapper.h"

int run_inference(Settings &opts, ModelTypes modelType, bool train = false, int16_t XAI = -1, float imgWeight = 0.5, bool save = true);
/*
* Function to gather information about the images to be passed for inference
* Calls lenet_inference
* 
* * credit for the file iterator:
*	 https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c 
* 
* Args:
*	opts: options for inference
*	modelType: chooses which model to run
*	train: determines whether to use the trained model or the defined model
*	XAI: fm to visualise (-1 for gradcam, -2 to skip)
*	save: whether to save the visualisations
*
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts, ModelTypes modelType, bool train = false, int16_t XAI = -1, float imgWeight = 0.5, bool save = true);
/*
* Function for getting results of inference
* 
* Args:
*	fImgs: container for information about the images
*	opts: options for inference
*	modelType: determine which model to use
*	train: determines whether to use the trained model or the defined model
*	XAI: fm to visualise (-1 for gradcam, -2 to skip)
*	save: whether to save the visualisations
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/



#endif