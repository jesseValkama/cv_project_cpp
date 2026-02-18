#include "models/resnet.h"

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdint.h>

#include "models/common.h"

namespace nn = torch::nn;
namespace ti = torch::indexing;

ResNetImpl::ResNetImpl(int imgsz, int64_t nCls, int64_t nc, bool fmvis)
{
	cb.in = nc;
	cache = fmvis;

	conv = ConvBlock(cb);
	maxPool = nn::MaxPool2d(nn::MaxPool2dOptions(mp.ks).stride(mp.s));
	layer1 = make_layer(rb1);
	layer2 = make_layer(rb2);
	layer3 = make_layer(rb3);
	layer4 = make_layer(rb4);
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

nn::Sequential ResNetImpl::make_layer(ResidualBlockParams &resParams)
{
	dsBlock = nullptr;
	if (resParams.firstStride != 1 || resParams.cb.in != resParams.cb.out)
	{
		ConvBlockParams tmp = { resParams.cb.in, resParams.cb.out, resParams.cb.ks, resParams.firstStride, resParams.cb.p };
		dsBlock = ConvBlock(tmp);
	}
	nn::Sequential layer = nn::Sequential();
	layer->push_back(ResidualBlock(resParams.cb, resParams.firstStride, dsBlock));
	
	for (int i = 1; i < resParams.n; ++i)
	{
		ConvBlockParams tmp = {resParams.cb.out, resParams.cb.out, resParams.cb.ks, resParams.cb.s, resParams.cb.p};
		layer->push_back(ResidualBlock(tmp, resParams.cb.s));
	}
	return layer;
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

std::optional<torch::Tensor> ResNetImpl::get_fm(int16_t fmi)
{
	if (fmi < -1 || fmi >= fm.size(1)) { std::cout << "Bad fm index" << "\n";  return std::nullopt; }
	if (!cache) { std::cout << "Init model for fm vis" << "\n"; return std::nullopt; }
	if (fmi != -1) { return fm.index({ti::Slice(), fmi, ti::Slice(), ti::Slice()}); }
	return fm;
}
