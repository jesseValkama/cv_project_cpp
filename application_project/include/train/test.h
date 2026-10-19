#ifndef TESTLOOP_H
#define TESTLOOP_H

#include <string>
#include <regex>

#include "functions/experiments_database.h"
#include "models/model_wrapper.h"
#include "metrics/classification_metrics.h"
#include "settings/settings.h"
#include "train/common.h"

template<typename Sequentialloader>
int test_loop(Sequentialloader &testloader, Settings &opts, const ModelTypes modelType, const std::string &datasetName,
	 train::Tracker &tracker, const bool useTrainedModel)
/*
	* The function to test LeNet, currently only tests the model from the last training
	* 
	* Args:
	*	testloader: dataloader for testing
	*	opts: settings for testing
	*	modelType: which model
	*	train: which model to test (trained or defined)
	* 
	* Returns:
	*	0: successful
	*	1: failed (logged to terminal)
*/
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::unique_ptr<ModelWrapper> modelWrapper = std::make_unique<ModelWrapper>(modelType, mnistOpts);
	std::string modelName = modelWrapper->get_name();
	std::cout << ANSI_MAGENTA << "Starting testing for " << modelName << ANSI_END << "\n";

	std::string fModel = useTrainedModel ? mnistOpts.workModel : mnistOpts.testModel;
	modelWrapper->load_weights(fModel);
	modelWrapper->to(opts.dev);
	modelWrapper->eval();
	MetricsContainer mc(mnistOpts.numOfClasses);

	torch::NoGradGuard no_grad;
	for (Batch &batch : testloader)
	{
		torch::Tensor imgs = batch.data.to(opts.dev, mnistOpts.async);
		torch::Tensor labels = batch.target.to(opts.dev, mnistOpts.async).view({ -1 });
		torch::Tensor outputs = modelWrapper->forward(imgs);
		calc_cm(labels, outputs, mc);
	}
	mc.print_cm();
	mc.calc_metrics();
	mc.print_metrics();
	mc.print_metrics(-2);
	AvgMetrics metrics = mc.get_metrics();

	std::string ans;
	std::cout << "Would you like to save the model?\nType name.pth to save the model, skip by not writing in the correct format" << std::endl;
	std::getline(std::cin, ans);
	std::regex rx(R"(^\w+\.pth$)");
	if (std::regex_match(ans, rx))
	{
		std::cout << "Saving the model" << std::endl;
		modelWrapper->save_weights(ans);
	}
	else
	{
		ans = "not_saved";
		std::cout << "Not saving the model" << std::endl;
	}
	int ret = insert_experiments(opts.databasePath.c_str(), 
		"experimentName", modelName.c_str(), datasetName.c_str(), tracker.trainTime.c_str(), tracker.epoch, ans.c_str(), metrics.at("recall"), 
									metrics.at("precision"), metrics.at("accuracy"), "optimiserName");
	return ret;
}

#endif // TESTLOOP_H