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
	* This is an implementation of ResNet18, which can be changed in the params below
	* 
	* Credits:
	* 
	* Official paper:
	*	https://arxiv.org/abs/1512.03385
	* Inspiration code:
	*	https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
	*	the padding values also come from that post 
	*/

	// the 1st is changed in the constructor method according to the input
	ConvBlockParams cb = {0,64,7,2,3};
	MaxPoolParams mp = {3,2,1};
	AvgPoolParams ap = {7,1,0};

	// the first stride is defined in the ResidualBlock with the pair
	// the changes in the depths are automatically handles
	ConvBlockParams rb1c = {64,64,3,1,1};
	ConvBlockParams rb2c = {64,128,3,1,1};
	ConvBlockParams rb3c = {128,256,3,1,1};
	ConvBlockParams rb4c = {256,512,3,1,1};
	
	// the pair is automatically operated (see make_layer), but you need to define the first stride
	ResidualBlockParams rb1 = {rb1c, 2, 1};
	ResidualBlockParams rb2 = {rb2c, 2, 2};
	ResidualBlockParams rb3 = {rb3c, 2, 2};
	ResidualBlockParams rb4 = {rb4c, 2, 2};

	ConvBlock conv{ nullptr };
	ConvBlock dsBlock{ nullptr };
	nn::MaxPool2d maxPool{ nullptr };
	nn::AvgPool2d avgPool{ nullptr };
	nn::Sequential layer4{ nullptr };
	nn::Sequential layer1{ nullptr };
	nn::Sequential layer2{ nullptr };
	nn::Sequential layer3{ nullptr };
	nn::Linear fc{ nullptr };

	bool cache = false;
	torch::Tensor fm;

	// if you add more blocks, you need to add them to this vector
	std::vector<blockTypes> layerParams = 
	{
		cb, mp, rb1, rb2, rb3, rb4, ap
	};
	
	ResNetImpl(int imgsz, int64_t nCls = 10, int64_t nc = 1, bool fmvis = false);
	nn::Sequential make_layer(ResidualBlockParams &p);
	torch::Tensor forward(torch::Tensor x);
	std::optional<torch::Tensor> get_fm(int16_t fmi);
};
TORCH_MODULE(ResNet);

#endif