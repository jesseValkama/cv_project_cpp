#include "lenet.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <torch/torch.h>

#include "common.h"
#include "../settings.h"

/*
* TODO:
* fix the conblock and make the model more modular
* also change the dynamic sizing computation to be a func
*/

LeNetImpl::LeNetImpl(int nc, int imgsz)
{
	ConvBlockParams cb1 = { 1, 6, 5, 1, 0, 6 };
	ConvBlockParams cb2 = { 6, 16, 5, 1, 0, 16 };
	
	//conv1 = ConvBlock(cb1);
	conv1 = nn::Conv2d(nn::Conv2dOptions(cb1.in, cb1.out, cb1.ks).stride(cb1.s).bias(false));
	bn1 = nn::BatchNorm2d(nn::BatchNorm2dOptions(cb1.bn));
	a1 = nn::ReLU();

	mp1 = nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2));

	conv2 = nn::Conv2d(nn::Conv2dOptions(cb2.in, cb2.out, cb2.ks).stride(cb2.s).bias(false));
	bn2 = nn::BatchNorm2d(nn::BatchNorm2dOptions(cb2.bn));
	a2 = nn::ReLU();
	//conv2 = ConvBlock(cb2);
	mp2 = nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(2));

	// computed only once, since the images are expected to be squares
	// make changes if the layers change!!!
	int64_t size = 0;
	size = (imgsz - cb1.ks + 2 * cb1.p) / (cb1.s) + 1;
	size = (size - 2) / (2) + 1;
	size = (size - cb2.ks + 2 * cb2.p) / (cb2.s) + 1;
	size = (size - 2) / (2) + 1;
	size = size * size * cb2.out;

	fc1 = nn::Linear(nn::LinearOptions(size,120));
	relu1 = nn::ReLU();
	fc2 = nn::Linear(nn::LinearOptions(120,84));
	relu2 = nn::ReLU(); // nn::ReLUOptions().inplace(true)
	fc3 = nn::Linear(nn::LinearOptions(84,nc));
	
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("a1", a1);

	//register_module("conv1", conv1);
	register_module("mp1", mp1);

	register_module("conv2", conv2);
	register_module("bn2", bn2);
	register_module("a2", a2);

	//register_module("conv2", conv2);
	register_module("mp2", mp2);

	register_module("fc1", fc1);
	register_module("relu1", relu1);
	register_module("fc2", fc2);
	register_module("relu2", relu2);
	register_module("fc3", fc3);
}

torch::Tensor LeNetImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = a1->forward(x);

	x = mp1->forward(x);

	x = conv2->forward(x);
	x = bn2->forward(x);
	x = a2->forward(x);

	x = mp2->forward(x);
	
	x = x.view({ x.size(0), -1 });

	x = fc1->forward(x);
	x = relu1->forward(x);
	x = fc2->forward(x);
	x = relu2->forward(x);
	x = fc3->forward(x);

	return x;
}
