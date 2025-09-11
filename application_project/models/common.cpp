#include "common.h"

#include <cstdlib>
#include <cstddef>
#include <stdint.h>
#include <variant>
#include <vector>

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

int64_t calc_size_reduction(int64_t sz, int64_t ks, int64_t p, int64_t s, int repeat)
{
	for (int i = 0; i < repeat; ++i)
	{
		sz = (sz - ks + 2 * p) / s + 1;
	}
	if (sz < 1)
	{
		std::cout << "The size of the output is < 1" << std::endl;
		std::abort();
	}
	return sz;
}

int64_t dynamic_fc(std::vector<blockTypes> &layerParams, int imgsz)
{
	int64_t sz = imgsz;
	ConvBlockParams cb;
	MaxPoolParams mp;
	AvgPoolParams ap;
	ResidualBlockParams rb;
	int64_t nc = 0;

	for (blockTypes &blockParams : layerParams)
	{
		ParamType type = static_cast<ParamType>(blockParams.index());
		switch (type)
		{
			case ParamType::ConvBlockType:
				cb = std::get<ConvBlockParams>(blockParams);
				sz = calc_size_reduction(sz, cb.ks, cb.p, cb.s);
				nc = cb.out;
				break;

			case ParamType::MaxPoolType:
				mp = std::get<MaxPoolParams>(blockParams);
				sz = calc_size_reduction(sz, mp.ks, mp.p, mp.s);
				break;
					
			case ParamType::AvgPoolType:
				ap = std::get<AvgPoolParams>(blockParams);
				sz = calc_size_reduction(sz, ap.ks, ap.p, ap.s);
				break;

			case ParamType::ResidualBlockType:
				rb = std::get<ResidualBlockParams>(blockParams);
				cb = rb.convBlockParams;
				sz = calc_size_reduction(sz, cb.ks, cb.p, rb.firstStride.second);
				sz = calc_size_reduction(sz, cb.ks, cb.p, cb.s);
				sz = calc_size_reduction(sz, cb.ks, cb.p, cb.s, rb.n - 1);
				nc = cb.out;
				break;

			default:
				std::cout << "Block type not added to dynamic_fc" << std::endl;
				std::abort();
		}
	}
	return sz * sz * nc;
}

ResidualBlockImpl::ResidualBlockImpl(ResidualBlockParams &p, ConvBlock downsample)
{
	params = p;
	conv1 = ConvBlock(p.convBlockParams);
	conv2 = ConvBlock(p.convBlockParams);
	dsBlock = downsample;
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
	if (dsBlock) 
	{
		dsBlock->forward(res);
	}
	return relu->forward(x);
}
