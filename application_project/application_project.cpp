#include "application_project.h"

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

#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[])
{
	/*
	* todo:
	* 
	* IMPLEMENT AMP!!
	* complete readme
	* add unit tests (does googletest even work with libtorch?)
	*/
	
	std::optional<std::unordered_map<std::string, int16_t>> args = handle_args(argc, argv);
	if (!args.has_value()) { return 1; }
	
	// todo: make fn for these with erro handling in handle_args with a tuple
	bool train = args->at("train") == 1 ? true : false;
	bool test = args->at("test") == 1 ? true : false;
	bool inference = args->at("inference") == 1 ? true : false;
	ModelTypes modelType = static_cast<ModelTypes>(args->at("model"));
	int16_t XAI = args->at("xai");
	DatasetTypes datasetType = static_cast<DatasetTypes>(args->at("dataset"));

	Settings opts(datasetType, modelType);
	int ret = 0;

	if (train || test)
	{
		ret = lenet_loop(opts, modelType, datasetType, train, test);
		if (ret != 0) { return ret; }
	}
	if (inference)
	{
		ret = run_inference(opts, modelType, train, XAI);
		if (ret != 0) { return ret; }
	}
	return 0;
}

