#include "resnet.h"

#include <torch/torch.h>

#include <stdint.h>

#include "common.h"

namespace nn = torch::nn;

ResNetImpl::ResNetImpl(int64_t nc)
{
	conv = ConvBlock(cb);
	maxPool = nn::MaxPool2d(nn::MaxPool2dOptions(mp.ks).stride(mp.s));
	layer1 = nn::Sequential();
	layer2 = nn::Sequential();
	layer3 = nn::Sequential();
	layer4 = nn::Sequential();
	avgPool = nn::AvgPool2d(nn::AvgPool2dOptions(ap.ks).stride(ap.s));

	int64_t size = 120;
	fc = nn::Linear(size, nc);

	register_module("conv", conv);
	register_module("maxPool", maxPool);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("avgPool", avgPool);
	register_module("fc", fc);
}

nn::Sequential ResNetImpl::make_layer(nn::Sequential layers, uint32_t n, ResidualBlockParams &resParams, ConvBlockParams &convParams)
{
	if (resParams.convBlockParams.s != 1 || cachedDepth != resParams.convBlockParams.out)
	{
		dsBlock = ConvBlock(convParams);
		resParams.ds = true;
		layers->push_back(ResidualBlock(resParams, dsBlock));
		dsBlock = nullptr;
		resParams.ds = false;
	}
	
	for (int i = 1; i < n; ++i)
	{
		layers->push_back(ResidualBlock(resParams, dsBlock));
	}
	return layers;
}

torch::Tensor ResNetImpl::forward(torch::Tensor x)
{
	x = conv->forward(x);
	x = maxPool->forward(x);
	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);
	x = avgPool->forward(x);
	x = fc->forward(x);
	return x;
}
