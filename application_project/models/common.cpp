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
	bn = nn::BatchNorm2d(nn::BatchNorm2dOptions(p.bn));
	relu = nn::ReLU();

	register_module("conv", conv);
	register_module("bn", bn);
	register_module("relu", relu);
}

torch::Tensor ConvBlockImpl::forward(torch::Tensor x)
{
	x = conv->forward(x);
	x = bn->forward(x);
	return relu->forward(x);
}

int dynamicFC(int imgsz, ConvBlockParams &cb1, ConvBlockParams &cb2)
{
	// computed only once, since the images are expected to be squares
	// make changes if the layers change!!!
	int size = 0;
	size = (imgsz - cb1.ks + 2 * cb1.p) / (cb1.s) + 1;
	size = (size - 2) / (2) + 1;
	size = (size - cb2.ks + 2 * cb2.p) / (cb2.s) + 1;
	size = (size - 2) / (2) + 1;
	size = size * size;
	return 0;
}
