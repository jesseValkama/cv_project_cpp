#include "application_project.h"

#include <iostream>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "settings/handle_args.h"
#include "settings/settings.h"
#include "datasets/data_helper.h"
#include "inference/inference.h"
#include "models/model_wrapper.h"
#include "train/run_loop.h"
#include "visualise/common.h"

int main(int argc, char *argv[])
{
	/*
	* todo:
	*
	* modular train/val/test/inf loops
	* cnn + vit hybrid (with distillation)
	* document include/database (oh boy will i like documenting half a year old code)
	* graphs
	* add unit tests (valgrind + sanitisers + googletest + ctest)
	* fix strides of structs
	* logger
	* fix project structure (when/if it becomes a problem)
	* implement proper inference with support for models trained with pytorch (python)
	* There is a problem with model weight imports (I could not load the model trained on main desktop on my lapotp)
	*/
	std::unordered_map<std::string, int16_t> modelConversionMap = {{"LeNet", LENET}, {"ResNet", RESNET}};
	std::unordered_map<std::string, int16_t> datasetConversionMap = {{"mnist", MNIST}, {"cifar10", CIFAR10}};
	std::unordered_map<std::string, int16_t> visualtaskConversionMap = {{"SKIP", SKIP}, {"Idx", IDX}, {"GradCAM", GRADCAM}};
	std::unordered_map<VisualiserTasks, cv::ColormapTypes> colourmap = {{IDX, cv::COLORMAP_VIRIDIS}, {GRADCAM, cv::COLORMAP_JET}};
	args::Parser parser = args::Parser();
	int ret = parser.add_arguments({args::ArgInfo("--train"), args::ArgInfo("--test"), args::ArgInfo("--inf"), 
									args::ArgInfo("--visualtask", visualtaskConversionMap), 
									args::ArgInfo("--model", modelConversionMap), 
									args::ArgInfo("--dataset", datasetConversionMap)});
	if (ret != 0)
	{
		return ret;
	}
	std::optional<Args> args = parser.parse_args(argc, argv);
	if (!args.has_value()) 
	{ 
		return 1; 
	}
	bool train = args->at("--train").value == 1;
	bool test = args->at("--test").value == 1;
	bool inference = args->at("--inf").value == 1;
	ModelTypes modelType = static_cast<ModelTypes>(args->at("--model").value);
	VisualiserTasks visualiserTask = static_cast<VisualiserTasks>(args->at("--visualtask").value);
	std::vector<fmvis::Visualisers> visualisers = {fmvis::Visualisers(visualiserTask, colourmap.at(visualiserTask), 
																	args->at("--visualtask").meta)};
	DatasetTypes datasetType = static_cast<DatasetTypes>(args->at("--dataset").value);
	Settings settings(datasetType, modelType);

	std::cout << "size of settings: " << sizeof(Settings) << std::endl;

	if (train || test)
	{
		ret = train::run_loop(settings, modelType, datasetType, train, test);
		if (ret != 0) 
		{ 
			return ret; 
		}
	}
	if (inference)
	{
		ret = inf::run_inference(settings, modelType, visualisers, train, 0.5, false);
		if (ret != 0) 
		{ 
			return ret; 
		}
	}
	return 0;
}

