#include "application_project.h"

#include <iostream>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "settings.h"
#include "datasets/data_helper.h"
#include "functions/handle_args.h"
#include "functions/inference.h"
#include "functions/train_lenet.h"
#include "models/model_wrapper.h"

int main(int argc, char *argv[])
{
	/*
	* todo:
	*
	* why does my laptop with cpu train lenet (10 epochs) in 40 seconds when 
	* my main pc with gpu train it in 2 minutes with the same settings?
	* there is no way linux runs code in debug mode that much faster, right?
	* managed to mess up my path in windows then trying to implement release verions of dependencies
	*
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
	std::optional<std::unordered_map<std::string, int16_t>> args = handle_args(argc, argv);
	if (!args.has_value()) 
	{ 
		return 1; 
	}
	bool train = args->at("train") == 1 ? true : false;
	bool test = args->at("test") == 1 ? true : false;
	bool inference = args->at("inference") == 1 ? true : false;
	ModelTypes modelType = static_cast<ModelTypes>(args->at("model"));
	int16_t XAI = args->at("xai");
	DatasetTypes datasetType = static_cast<DatasetTypes>(args->at("dataset"));
	Settings settings(datasetType, modelType);
	int ret = 0;

	std::cout << "size of settings: " << sizeof(Settings) << std::endl;

	if (train || test)
	{
		ret = train::run_loop(settings, modelType, datasetType, train, test);
		if (ret != 0) { return ret; }
	}
	if (inference)
	{
		ret = run_inference(settings, modelType, train, XAI, 0.5, false);
		if (ret != 0) { return ret; }
	}
	return 0;
}

