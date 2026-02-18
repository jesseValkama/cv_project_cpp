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

#include <sqlite3.h>
#include "functions/database.h"

int main(int argc, char *argv[])
{
	/*
	* todo:
	* 
	* sqlite3 database TODO: finalise statements without executing for error handling
	* graphs
	* improve aug
	* vit
	* add unit tests (does googletest even work with libtorch?) and probably github actions too
	* 
	* improve readme
	*/
	std::optional<std::unordered_map<std::string, int16_t>> args = handle_args(argc, argv);
	if (!args.has_value()) { return 1; }
	bool train = args->at("train") == 1 ? true : false;
	bool test = args->at("test") == 1 ? true : false;
	bool inference = args->at("inference") == 1 ? true : false;
	ModelTypes modelType = static_cast<ModelTypes>(args->at("model"));
	int16_t XAI = args->at("xai");
	DatasetTypes datasetType = static_cast<DatasetTypes>(args->at("dataset"));
	Settings settings(datasetType, modelType);
	int ret = 0;
	
	// tmp code
	sqlite3 *database = nullptr;
	const char *filename = "D:/self-studies/application_project/application_project/test.db";
	ret = db::open(filename, &database);
	if (ret != 0) { return ret; }
	//ret = db::create_experiments(database);
	DatabaseConstructor databaseConstructor = db::get_database_constructor("experiment5", "resnet", "time", 80, "filename", 90.0, 90.0, 90.0, "adamw", settings);
	DatabaseMap databaseMap(std::move(databaseConstructor));
	ret = databaseMap.topoSort();
	if (ret != 0) { return ret; }
	ret = db::insert_experiments(database, databaseMap);
	if (ret != 0) { return ret; }
	ret = db::close(database);
	if (ret != 0) { return ret; }

	if (train || test)
	{
		ret = lenet_loop(settings, modelType, datasetType, train, test);
		if (ret != 0) { return ret; }
	}
	if (inference)
	{
		ret = run_inference(settings, modelType, train, XAI, 0.5, false);
		if (ret != 0) { return ret; }
	}
	return 0;
}

