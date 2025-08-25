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

struct MaxPoolParams
{
	int64_t ks;
	int64_t s;
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
	torch::Tensor forward(torch::Tensor x);

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

#endif