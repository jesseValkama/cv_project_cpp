#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

#include <stdint.h>

#include "common.h"

namespace nn = torch::nn;

struct ResNetImpl : torch::nn::Module
{
	int p = 0; // tmp
	ConvBlockParams cb = {1,64,7,2,p};
	MaxPoolParams mp = {3,2};
	AvgPoolParams ap = {7,1};

	// stride only defines the stride of the first conv in a layer
	ConvBlockParams rp1c = {64,64,3,1,p};
	ConvBlockParams rp2c = {64,128,3,2,p};
	ConvBlockParams rp3c = {128,256,3,2,p};
	ConvBlockParams rp4c = {256,512,3,2,p};

	ResidualBlockParams rp1 = {rp1c, false};
	ResidualBlockParams rp2 = {rp2c, false};
	ResidualBlockParams rp3 = {rp3c, false};
	ResidualBlockParams rp4 = {rp4c, false};

	ConvBlock conv{ nullptr };
	ConvBlock dsBlock{ nullptr };
	nn::MaxPool2d maxPool{ nullptr };
	nn::AvgPool2d avgPool{ nullptr };
	nn::Sequential layer4{ nullptr };
	nn::Sequential layer1{ nullptr };
	nn::Sequential layer2{ nullptr };
	nn::Sequential layer3{ nullptr };
	nn::Linear fc{ nullptr };
	int64_t cachedDepth = 0;
	
	ResNetImpl(int64_t nc);
	nn::Sequential make_layer(nn::Sequential layers, uint32_t n, ResidualBlockParams &p, ConvBlockParams &convParams);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResNet);

#endif