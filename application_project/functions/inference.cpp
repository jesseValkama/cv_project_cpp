#include "inference.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../models/lenet.h"
#include "../settings.h"

namespace fs = std::filesystem;

/*
* credit for the file iterator:
* https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c 
*/

int run_inference(Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	std::vector<std::string> fImgs;
	for (const auto& entry : fs::directory_iterator(mnistOpts.inferenceDataPath))
	{
		fImgs.emplace_back(entry.path().string());
	}
	int ret = 0;
	ret = lenet_inference(fImgs, opts);
	if (ret != 0) { return ret; }
	return 0;
}

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts)
{
	std::cout << "starting inference" << "\n";
	
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgsz, true);
	std::string fModel = mnistOpts.savepath + "/" + mnistOpts.inferenceModel + ".pth";
	torch::load(model, fModel);
	model->to(opts.dev);
	model->eval();
	int64_t p = 0, l = 0;
	
	cv::namedWindow("fmvis", cv::WINDOW_AUTOSIZE);
	int n = fImgs.size();
	torch::NoGradGuard no_grad;
	for (int i = 0; i < n; ++i)
	{
		std::optional<cv::Mat> img = load_png_greyscale_img(fImgs[i], mnistOpts.imgresz);
		if (!img.has_value()) { return 1; }
		torch::Tensor timg = greyscale2Tensor(img.value(), mnistOpts.imgresz);
		timg = timg.to(opts.dev);
		timg = timg.unsqueeze(0);

		torch::Tensor logits = model->forward(timg);
		torch::Tensor preds = torch::softmax(logits, 1);
		preds = preds.to(torch::kFloat);

		torch::Tensor xi = torch::argmax(preds[0]);
		torch::Tensor prob = preds[0][xi];
		
		xi = xi.to(torch::kCPU);
		prob = prob.to(torch::kCPU, torch::kFloat);

		l = xi.item<int64_t>();
		p = prob.item<float>(); // todo: fix the probabilities, this makes no sense
		std::cout << std::fixed << std::setprecision(6) << "The output is: " << l << ", with the probability of: " << p << std::endl;
		
		std::optional<torch::Tensor> tfm = model->get_fm(0);
		if (!tfm.has_value()) { return 2; }
		visualise_fm(tfm.value());
	}
	cv::destroyWindow("fmvis");
	return 0;
}

void visualise_fm(torch::Tensor tfm)
{
	tfm.squeeze_(0);
	cv::Mat fm = Tensor2greyscale(tfm, true);
	cv::Mat cm;
	cv::applyColorMap(fm, cm, cv::COLORMAP_DEEPGREEN);
	cv::resize(cm, cm, cv::Size(100, 100));
	
	cv::imshow("fmvis", cm);
	cv::waitKey(0);
}