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
};

struct MaxPoolParams
{
	int64_t ks;
	int64_t s;
	int64_t p = 0;
};

struct AvgPoolParams
{
	int64_t ks;
	int64_t s;
	int64_t p = 0;
};

struct ResidualBlockParams
{
	ConvBlockParams convBlockParams;
	bool ds = false;
	int64_t n = 0;
};


struct ConvBlockImpl : torch::nn::Module
{
	/*
	* Convolution block used by LeNet (custom version):
	*	convolution 2d
	*	batch normalisation 2d
	*	relu
	* 
	* Attributes:
	*	ptrs: ptrs to blocks
	*/
	ConvBlockImpl(const ConvBlockParams& p);
	torch::Tensor forward(torch::Tensor x, bool bRelu = true);
	/*
	* Method to forward pass the img
	* 
	* Args:
	*	x: input as a Tensor
	*	bRelu: bool whether to use relu
	* 
	* Returns:
	*	Tensor: the output feature map
	*/

	nn::Conv2d conv{ nullptr };
	nn::BatchNorm2d bn{ nullptr };
	nn::ReLU relu{ nullptr };
};

TORCH_MODULE(ConvBlock);

int64_t dynamicFC(int imgsz, ConvBlockParams &cb1, MaxPoolParams &mp1, ConvBlockParams &cb2, MaxPoolParams &mp2);
/*
* Function used to calculate dynamic input size for the fully connected layer for classification
* 
* Args:
*	imgsz: the original size of the imgs
*	cb1: params for the 1st convblock
*	mp1: params for the 1st maxpool
*	cb2: params for the 2nd convblock
*	mp2: params for the 2nd maxpool
* 
* Returns:
*	size: calculated as w * h * n of filters 
*/

struct ResidualBlockImpl : torch::nn::Module
{
	ConvBlock conv1{ nullptr };
	ConvBlock conv2{ nullptr };
	ConvBlock dsBlock{ nullptr };
	nn::ReLU relu{ nullptr };
	ResidualBlockParams params;

	ResidualBlockImpl(ResidualBlockParams &p, ConvBlock &downsample);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResidualBlock);

#endif