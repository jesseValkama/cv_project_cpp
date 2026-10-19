#include "inference/inference.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>

#include "datasets/loader_funcs.h"
#include "datasets/inference.h"
#include "models/model_wrapper.h"
#include "settings/settings.h"

namespace fs = std::filesystem;

int inf::run_inference(const Settings &opts, const ModelTypes modelType, const std::vector<fmvis::Visualisers> &visualisers, 
	const bool train, const float imgWeight, const bool save)
{
	DatasetOpts dsOpts = opts.mnistOpts;
	std::vector<std::string> imgPaths;
	for (const auto& entry : fs::directory_iterator(dsOpts.inferenceDataPath))
	{
		imgPaths.emplace_back(entry.path().string());
	}
	auto infDataset = InferenceDataset(imgPaths, dsOpts, dsOpts.async)
		.map(torch::data::transforms::Normalize<>(
			dsOpts.mean, dsOpts.stdev))
		.map(torch::data::transforms::Stack<>());
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
	(
		std::move(infDataset),
		torch::data::DataLoaderOptions().batch_size(dsOpts.testBS).workers(dsOpts.numWorkers)
	);
	int ret = inference(*dataloader, opts, modelType, visualisers, train, save);
	if (ret != 0) { return ret; }
	return 0;
}