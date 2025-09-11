#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

#include <optional>
#include <stdint.h>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"

namespace nn = torch::nn;

struct ResNetImpl : torch::nn::Module
{
	/*
	* Credits:
	* 
	* Official paper:
	*	https://arxiv.org/abs/1512.03385
	* Inspiration code:
	*	https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
	* 
	* The model is still not working
	*/

	int p = 0; // tmp
	// the 1st is changed in the constructor method according to the input
	ConvBlockParams cb = {0,64,7,2,p};
	MaxPoolParams mp = {3,2};
	AvgPoolParams ap = {7,1};

	// the first stride is defined in the ResidualBlock with the pair
	ConvBlockParams rp1c = {64,64,3,1,p};
	ConvBlockParams rp2c = {64,128,3,1,p};
	ConvBlockParams rp3c = {128,256,3,1,p};
	ConvBlockParams rp4c = {256,512,3,1,p};
	
	// the pair is automatically operated (see make_layer), but you need to define the first stride
	ResidualBlockParams rp1 = {rp1c, 3, std::make_pair(false, 2)};
	ResidualBlockParams rp2 = {rp2c, 4, std::make_pair(false, 2)};
	ResidualBlockParams rp3 = {rp3c, 6, std::make_pair(false, 2)};
	ResidualBlockParams rp4 = {rp4c, 3, std::make_pair(false, 2)};

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
	bool cache = false;
	torch::Tensor fm;

	// if you add more blocks, you need to add them to this vector
	std::vector<blockTypes> layerParams = 
	{
		cb, mp, rp1, rp2, rp3, rp4, ap
	};
	
	ResNetImpl(int imgsz, int64_t nCls = 10, int64_t nc = 1, bool fmvis = false);
	nn::Sequential make_layer(nn::Sequential layers, ResidualBlockParams &p, ConvBlockParams &convParams);
	torch::Tensor forward(torch::Tensor x);
	std::optional<torch::Tensor> get_fm(int fmi);
};
TORCH_MODULE(ResNet);

#endif