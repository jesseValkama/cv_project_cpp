#ifndef CONVBLOCK_H
#define CONVBLOCK_H

#include <cstddef>
#include <stdint.h>
#include <torch/torch.h>

#include "../settings.h"

namespace nn = torch::nn;

struct ConvBlockParams
{
	int64_t in;
	int64_t out;
	int64_t ks;
	int64_t s;
	int64_t p;
	int64_t bn;
};

struct ConvBlockImpl : torch::nn::Module
{
	ConvBlockImpl(const ConvBlockParams &p);
	
	torch::Tensor forward(torch::Tensor x);

	nn::Conv2d conv{ nullptr };
	nn::BatchNorm2d bn{ nullptr };
	nn::ReLU relu{ nullptr };
};

TORCH_MODULE(ConvBlock);

#endif