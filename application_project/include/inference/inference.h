#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <optional>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "datasets/data_helper.h"
#include "models/model_wrapper.h"
#include "settings/settings.h"
#include "visualise/feature_map_visualiser.h"

namespace inf
{

// TODO: rm
struct Tracker
{
	std::unordered_map<std::string, torch::Tensor> process_logits(torch::Tensor &logits)
	{
		torch::Tensor preds = torch::softmax(logits, 1).to(torch::kCPU, torch::kFloat);
		auto [probs, idxs] = torch::max(preds, 1);
		return {{"preds", preds}, {"probs", probs}, {"idxs", idxs}, {"logits", logits}};
	};
};

int run_inference(const Settings &opts, const ModelTypes modelType, const std::vector<fmvis::Visualisers> &visualisers, 
	const bool train = false, const float imgWeight = 0.5, const bool save = true);
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
*   visualisers: options for featuremap visualisations
*	imgWeight: scaling value (for the input img) when merging input img and feature map visualisation
*	save: whether to save the visualisations
*
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

} // end of inf

template <typename Dataloader>
int inference(Dataloader &dataloader, const Settings &opts, const ModelTypes modelType, 
	const std::vector<fmvis::Visualisers> &visualisers, const bool train = false, const float imgWeight = 0.5, 
	const bool save = true)
/*
* Inference function with support for feature map visualisations e.g., gradcam
*
* Args:
*	dataloader: dataloader
*	opts: options for inference
*	modelType: determine which model to use
*	train: determines whether to use the trained model or the defined model
*	XAI: fm to visualise (-1 for gradcam, -2 to skip)
*	imgWeight: scaling value (for the input img) when merging input img and feature map visualisation
*	save: whether to save the visualisations
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/
{
	DatasetOpts dsOpts = opts.mnistOpts;
	std::unique_ptr<ModelWrapper> modelWrapper = std::make_unique<ModelWrapper>(modelType, dsOpts, true);
	std::string fModel = train ? dsOpts.workModel : dsOpts.inferenceModel;
	modelWrapper->load_weights(fModel);
	modelWrapper->to(opts.dev);
	modelWrapper->eval();
	int ret = 0;
	cv::namedWindow("fmvis", cv::WINDOW_AUTOSIZE);
	inf::Tracker tracker = inf::Tracker();
	fmvis::FeatureMapVisualiserOptions fmopts = fmvis::FeatureMapVisualiserOptions(0.5, std::nullopt); // todo: fix to match torch patterns
	fmvis::FeatureMapVisualiser featureMapVisualiser = fmvis::FeatureMapVisualiser(visualisers, fmopts, dsOpts);

	std::cout << "Starting inference for " + modelWrapper->get_name() << std::endl;
	for (Batch &batch : dataloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, dsOpts.async);
		// at::autocast::set_autocast_enabled(opts.devType, opts.amp);
		// at::autocast::set_autocast_dtype(opts.devType, opts.ampDtype);
		torch::Tensor logits = modelWrapper->forward(imgs);
		// at::autocast::set_autocast_enabled(opts.devType, false);
		std::optional<torch::Tensor> fm = modelWrapper->get_fm(visualisers[0].task == GRADCAM ? -1 : *visualisers[0].idx); // TODO: fix
		std::unordered_map<std::string, torch::Tensor> outputs = tracker.process_logits(logits);
		if (fm.has_value())
		{
			featureMapVisualiser.visualise(imgs, *fm, outputs);
		}
	}
	cv::destroyWindow("fmvis");
	return 0;
}

#endif // INFERENCE_H