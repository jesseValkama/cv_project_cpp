#ifndef FEATUREMAPVISUALISER_H
#define FEATUREMAPVISUALISER_H

#include <torch/torch.h>

#include <stdint.h>
#include <unordered_map>
#include <vector>

#include "settings/settings.h"
#include "visualise/common.h"

namespace fmvis
{

class FeatureMapVisualiser
{
/*
*   Class to visualise feature maps
*/
private:
    std::vector<Visualisers> tasks;
    FeatureMapVisualiserOptions fmopts;
    DatasetOpts dsOpts;
public:
    FeatureMapVisualiser(const std::vector<Visualisers> &tasks, const FeatureMapVisualiserOptions &fmopts, const DatasetOpts &dsOpts) 
            : tasks(tasks), fmopts(fmopts), dsOpts(dsOpts) {};
    int visualise(torch::Tensor &imgs, torch::Tensor &featureMaps, std::unordered_map<std::string, torch::Tensor> &outputs);
    /*
    * Method for visualising feature maps
    *
    * Args
    *   imgs: input imgs as a tensor
    *   featureMaps: intermediate feature maps to visualise as a tensor (fms before gap recommended for gradcam; check the paper)
    *   outputs: model outputs
    */
};

} // end of fmvis


int visualise_fm(torch::Tensor &imgs, torch::Tensor &featureMaps, std::unordered_map<std::string, torch::Tensor> &outputs,
	DatasetOpts &dsOpts, fmvis::FeatureMapVisualiserOptions &fmopts);
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
*	savepath: the path to save the img to (nullopt to not save) 
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
* 
* Saves:
*	the feature map visualisation if save
*/

#endif // FEATUREMAPVISUALISER_H