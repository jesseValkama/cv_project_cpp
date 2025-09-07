#include "inference.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../functions/gradcam.h"
#include "../models/lenet.h"
#include "../settings.h"
#include "visualise.h"

namespace fs = std::filesystem;

/*
* credit for the file iterator:
* https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c 
*/

int run_inference(Settings &opts, int idx)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	std::vector<std::string> fImgs;
	for (const auto& entry : fs::directory_iterator(mnistOpts.inferenceDataPath))
	{
		fImgs.emplace_back(entry.path().string());
	}
	int ret = 0;
	ret = lenet_inference(fImgs, opts, idx);
	if (ret != 0) { return ret; }
	return 0;
}

int lenet_inference(std::vector<std::string> &fImgs, Settings &opts, int idx)
{
	std::cout << "starting inference" << "\n";
	
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgsz, true);
	std::string fModel = mnistOpts.savepath + "/" + mnistOpts.inferenceModel + ".pth";
	torch::load(model, fModel);
	model->to(opts.dev);
	model->eval();
	int64_t p = 0, l = 0;
	int ret = 0;
	
	cv::namedWindow("fmvis", cv::WINDOW_AUTOSIZE);
	int n = fImgs.size();
	//torch::NoGradGuard no_grad;
	for (int i = 0; i < n; ++i)
	{
		std::optional<cv::Mat> img = load_png_greyscale_img(fImgs[i], mnistOpts.imgresz);
		if (!img.has_value()) { return 1; }
		torch::Tensor timg = greyscale2Tensor(img.value(), mnistOpts.imgresz);
		timg = timg.to(opts.dev).unsqueeze_(0);

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
		
		if (idx == -2) { continue; }

		std::optional<torch::Tensor> tfm = model->get_fm(idx);
		if (!tfm.has_value()) { return 2; }

		if (idx == -1)
		{
			ret = gradcam(logits[0][xi], tfm.value(), timg);
			if (ret != 0) { return ret; }
		}
		else
		{
			ret = visualise_fm(tfm.value().squeeze_(0), timg, cv::COLORMAP_DEEPGREEN);
			if (ret != 0) { return ret; }
		}
	}
	cv::destroyWindow("fmvis");
	return 0;
}
