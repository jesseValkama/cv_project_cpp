#include "resnet.h"

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdint.h>

#include "common.h"

namespace nn = torch::nn;
namespace ti = torch::indexing;

ResNetImpl::ResNetImpl(int imgsz, int64_t nCls, int64_t nc, bool fmvis)
{
	cb.in = nc;
	cache = fmvis;

	conv = ConvBlock(cb);
	maxPool = nn::MaxPool2d(nn::MaxPool2dOptions(mp.ks).stride(mp.s));
	layer1 = nn::Sequential();
	layer2 = nn::Sequential();
	layer3 = nn::Sequential();
	layer4 = nn::Sequential();
	avgPool = nn::AvgPool2d(nn::AvgPool2dOptions(ap.ks).stride(ap.s));

	int64_t sz = dynamic_fc(layerParams, imgsz);
	layerParams.clear();

	fc = nn::Linear(sz, nCls);

	register_module("conv", conv);
	register_module("maxPool", maxPool);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("avgPool", avgPool);
	register_module("fc", fc);
}

nn::Sequential ResNetImpl::make_layer(nn::Sequential layers, ResidualBlockParams &resParams, ConvBlockParams &convParams)
{
	if (resParams.convBlockParams.s != 1 || cachedDepth != resParams.convBlockParams.out)
	{
		dsBlock = ConvBlock(convParams);
		resParams.firstStride.first = true;
		layers->push_back(ResidualBlock(resParams, dsBlock));
		resParams.firstStride.first = false;
		dsBlock = nullptr;
	}
	
	for (int i = 1; i < resParams.n; ++i)
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

	if (cache && !is_training())
	{
		x.retain_grad(); // unoptimised, always retains even if doing normal fm vis
		fm = x;
	}

	x = avgPool->forward(x);
	x = x.view({ x.size(0), -1 });
	x = fc->forward(x);
	return x;
}

std::optional<torch::Tensor> ResNetImpl::get_fm(int fmi)
{
	if (fmi < -1 || fmi >= fm.size(1)) { std::cout << "Bad fm index" << "\n";  return std::nullopt; }
	if (!cache) { std::cout << "Init model for fm vis" << "\n"; return std::nullopt; }
	if (fmi != -1) { return fm.index({ti::Slice(), fmi, ti::Slice(), ti::Slice()}); }
	return fm;
}
