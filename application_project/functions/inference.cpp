#include "inference.h"

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

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../functions/gradcam.h"
#include "../models/model_wrapper.h"
#include "../settings.h"
#include "visualise.h"

namespace fs = std::filesystem;

int run_inference(Settings &opts, ModelTypes modelType, bool train, int16_t XAI)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::vector<std::string> fImgs;
	for (const auto& entry : fs::directory_iterator(mnistOpts.inferenceDataPath))
	{
		fImgs.emplace_back(entry.path().string());
	}
	int ret = 0;
	ret = lenet_inference(fImgs, opts, modelType, train, XAI);
	if (ret != 0) { return ret; }
	return 0;
}

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts, ModelTypes modelType, bool train, int16_t XAI)
{
	DatasetOpts mnistOpts = opts.mnistOpts;
	std::unique_ptr<ModelWrapper> modelWrapper = std::make_unique<ModelWrapper>(modelType, mnistOpts, true);
	std::string fModel = train ? mnistOpts.workModel : mnistOpts.inferenceModel;
	modelWrapper->load_weights(fModel);
	modelWrapper->to(opts.dev, true);
	modelWrapper->eval();
	int64_t l = 0;
	double p = 0.0;
	int ret = 0;
	
	cv::namedWindow("fmvis", cv::WINDOW_AUTOSIZE);
	int n = fImgs.size();

	std::cout << "Starting inference for " + modelWrapper->get_name() << std::endl;
	//torch::NoGradGuard no_grad;
	for (int i = 0; i < n; ++i)
	{
		std::optional<cv::Mat> img = load_png_greyscale_img(fImgs[i], mnistOpts.imgresz);
		if (!img.has_value()) { return 1; }
		torch::Tensor timg = greyscale2Tensor(img.value(), mnistOpts.imgresz);
		timg = timg.to(opts.dev).unsqueeze_(0);

		torch::Tensor logits = modelWrapper->forward(timg);
		logits = logits.to(torch::kFloat64);
		torch::Tensor preds = torch::softmax(logits, 1);
		preds = preds.to(torch::kFloat);

		torch::Tensor xi = torch::argmax(preds[0]);
		torch::Tensor prob = preds[0][xi];
		
		xi = xi.to(torch::kCPU);
		prob = prob.to(torch::kCPU, torch::kFloat);

		l = xi.item<int64_t>();
		p = prob.item<double>();
		std::cout << std::fixed << std::setprecision(6) << "The output is: " << l << ", with the probability of: " << p << std::endl;
		
		if (XAI == -2) { continue; }

		std::optional<torch::Tensor> tfm = modelWrapper->get_fm(XAI);
		if (!tfm.has_value()) { return 2; }

		if (XAI == -1)
		{
			ret = gradcam(logits[0][xi], tfm.value(), timg, l, p);
			if (ret != 0) { return ret; }
		}
		else
		{
			ret = visualise_fm(tfm.value().squeeze_(0), timg, l, p, cv::COLORMAP_DEEPGREEN);
			if (ret != 0) { return ret; }
		}
	}
	cv::destroyWindow("fmvis");
	return 0;
}
