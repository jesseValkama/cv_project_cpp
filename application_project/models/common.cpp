#include "common.h"

#include <cstddef>
#include <stdint.h>
#include <torch/torch.h>
#include "../settings.h"

ConvBlockImpl::ConvBlockImpl(const ConvBlockParams &p)
{
	assert(
		p.in >= 0 &&
		p.out >= 0 &&
		p.ks >= 0 &&
		p.s >= 0 &&
		p.p >= 0
	);
	conv = nn::Conv2d(nn::Conv2dOptions(p.in, p.out, p.ks).stride(p.s).bias(false));
	bn = nn::BatchNorm2d(nn::BatchNorm2dOptions(p.out));
	relu = nn::ReLU();

	register_module("conv", conv);
	register_module("bn", bn);
	register_module("relu", relu);
}

torch::Tensor ConvBlockImpl::forward(torch::Tensor x, bool bRelu)
{
	x = conv->forward(x);
	x = bn->forward(x);
	return bRelu ? relu->forward(x) : x;
}

int64_t dynamicFC(int imgsz, ConvBlockParams &cb1, MaxPoolParams &mp1, ConvBlockParams &cb2, MaxPoolParams &mp2)
{
	// computed only once, since the images are expected to be squares
	// make changes if the layers change!!!
	int64_t size = 0;
	size = (imgsz - cb1.ks + 2 * cb1.p) / (cb1.s) + 1;
	size = (size - mp1.ks) / (mp1.s) + 1;
	size = (size - cb2.ks + 2 * cb2.p) / (cb2.s) + 1;
	size = (size - mp2.ks) / (mp2.s) + 1;
	return size * size * cb2.out;
}

ResidualBlockImpl::ResidualBlockImpl(ResidualBlockParams &p, ConvBlock &downsample)
{
	params = p;
	conv1 = ConvBlock(p.convBlockParams);
	conv2 = ConvBlock(p.convBlockParams);
	if (params.ds) { dsBlock = downsample; }
	relu = nn::ReLU();

	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("dsBlock", dsBlock);
	register_module("relu", relu);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x)
{
	torch::Tensor res = x;
	x = conv1->forward(x);
	x = conv2->forward(x, false);
	x += res;
	if (params.ds) 
	{
		dsBlock->forward(res);
	}
	return relu->forward(x);
}
