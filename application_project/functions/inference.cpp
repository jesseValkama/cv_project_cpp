#include "inference.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
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

void run_inference(Settings &opts)
{
	MnistOpts mnistOpts = opts.mnistOpts;
	std::vector<std::string> fImgs;
	for (const auto& entry : fs::directory_iterator(mnistOpts.inferenceDataPath))
	{
		fImgs.push_back(entry.path().string());
	}
	
	lenet_inference(fImgs, opts);
}

void lenet_inference(std::vector<std::string> &fImgs, Settings &opts)
{
	std::cout << "starting inference" << "\n";
	
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgsz, true);
	std::string fModel = mnistOpts.savepath + "/" + mnistOpts.inferenceModel + ".pth";
	torch::load(model, fModel);
	model->to(opts.dev);
	model->eval();
	int64_t p = 0, l = 0;
	
	int n = fImgs.size();
	torch::NoGradGuard no_grad;
	for (int i = 0; i < n; ++i)
	{
		cv::Mat img = load_png_greyscale_img(fImgs[i], mnistOpts.imgresz);
		torch::Tensor timg = greyscale2Tensor(img, mnistOpts.imgresz);
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

		torch::Tensor tfm = model->get_fm(0); // todo visualise
		tfm.squeeze_(0);
		cv::Mat fm = Tensor2greyscale(tfm, true);
		cv::Mat cm;
		cv::applyColorMap(fm, cm, cv::COLORMAP_DEEPGREEN);
		cv::resize(cm, cm, cv::Size(100, 100));
		
		/*cv::namedWindow("fmvis", cv::WINDOW_AUTOSIZE);
		cv::imshow("fmvis", cm);
		cv::waitKey(0);
		cv::destroyWindow("fmvis");*/
	}
}