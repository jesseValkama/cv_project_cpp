#include "lenet.h"

#include <cassert>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <stdint.h>
#include <torch/torch.h>

#include "common.h"
#include "../settings.h"

LeNetImpl::LeNetImpl(int nc, int imgsz, bool fmvis)
{
	cache = fmvis;

	conv1 = ConvBlock(cb1);
	mPool1 = nn::MaxPool2d(nn::MaxPool2dOptions(mp1.ks).stride(mp1.s));
	conv2 = ConvBlock(cb2);
	mPool2 = nn::MaxPool2d(nn::MaxPool2dOptions(mp2.ks).stride(mp2.s));
	
	int64_t sz = dynamicFC(imgsz, cb1, mp1, cb2, mp2);
	fc1 = nn::Linear(nn::LinearOptions(sz,120));
	relu1 = nn::ReLU();
	fc2 = nn::Linear(nn::LinearOptions(120,84));
	relu2 = nn::ReLU();
	fc3 = nn::Linear(nn::LinearOptions(84,nc));
	
	register_module("conv1", conv1);
	register_module("mPool1", mPool1);
	register_module("conv2", conv2);
	register_module("mPool2", mPool2);

	register_module("fc1", fc1);
	register_module("relu1", relu1);
	register_module("fc2", fc2);
	register_module("relu2", relu2);
	register_module("fc3", fc3);
}

torch::Tensor LeNetImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = mPool1->forward(x);
	x = conv2->forward(x);
	x = mPool2->forward(x);

	if (cache && !is_training())
	{
		fm = x.clone();
	}
	
	x = x.view({ x.size(0), -1 });
	x = fc1->forward(x);
	x = relu1->forward(x);
	x = fc2->forward(x);
	x = relu2->forward(x);
	x = fc3->forward(x);

	return x;
}

std::optional<torch::Tensor> LeNetImpl::get_fm(int fmi)
{
	if (!cache)
	{
		return std::nullopt;
	}
	if (fmi != -1) { return fm.index({ti::Slice(), fmi, ti::Slice(), ti::Slice()}); }
	return fm;
}
