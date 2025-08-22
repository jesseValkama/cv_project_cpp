#include "inference.h"

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <string>

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
	for (const auto& entry : fs::directory_iterator(mnistOpts.inferenceDataPath))
	{
		// place in tensor
		std::cout << entry.path() << std::endl;
	}

	// run lenet_inference
	return 0;
}

int lenet_inference(torch::Tensor imgs, Settings &opts)
{
	std::cout << "starting inference" << "\n";
	
	MnistOpts mnistOpts = opts.mnistOpts;
	LeNet model(mnistOpts.numOfChannels, mnistOpts.imgsz);
	model->to(opts.dev);
	model->eval();
	int64_t p = 0, l = 0;
	
	auto n = imgs.sizes(); // todo: typecast to int
	torch::NoGradGuard no_grad;
	for (int i = 0; i < n[0]; ++i)
	{
		torch::Tensor img = imgs[i];
		torch::Tensor logits = model->forward(img);
		torch::Tensor preds = torch::softmax(logits, 0);

		torch::Tensor xi = torch::argmax(preds);
		torch::Tensor prob = preds[xi];

		xi = xi.to(torch::kCPU);
		prob = prob.to(torch::kCPU);

		l = xi.item<int64_t>();
		p = prob.item<int64_t>();
	}
	return 0;
}