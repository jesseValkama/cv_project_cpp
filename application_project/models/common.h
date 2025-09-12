#ifndef CONVBLOCK_H
#define CONVBLOCK_H

#include <cstddef>
#include <stdint.h>
#include <utility>
#include <variant>
#include <vector>

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
	ConvBlockParams cb;
	int64_t n = 0;
	int64_t firstStride = 0;
};

// you need to update these if you add more custom blocks
typedef std::variant<ConvBlockParams, MaxPoolParams, AvgPoolParams, ResidualBlockParams> blockTypes;
enum ParamType
{
	ConvBlockType = 0,
	MaxPoolType = 1,
	AvgPoolType = 2,
	ResidualBlockType = 3
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

int64_t calc_size_reduction(int64_t sz, int64_t ks, int64_t p, int64_t s, int repeat = 1);
/*
* Function used to calculate an individual reduction of an image or a feature map after
* passing through a block like conv or pool
* 
* Args:
*	sz: the original size
*	ks: kernel size
*	p: padding
*	s: stride
*	repeat: how many times to repeat
* 
* Returns
*	int64_t: the output size
* 
* Abort:
*	if output < 1
* 
*/

int64_t dynamic_fc(std::vector<blockTypes> &layerParams, int imgsz);
/*
* Function used to calculate dynamic input size for the fully connected layer for classification
* Pooling operations are assumed to keep the number of channels the same
* Function uses std::variant underneath the hood (layerParams)
*	-> might need optimisations if heavier blocks are added, since libtorch forces int64_t
* 
* Args:
*	layerParams: vector of block params such as ConvBlockParams, ResidualBlockParams
* 
* Returns:
*	int64_t: the size as w * w * nc (presumes a square img)
* 
* Abort:
*	if output < 1
*/

struct ResidualBlockImpl : torch::nn::Module
{
	ConvBlock conv1{ nullptr };
	ConvBlock conv2{ nullptr };
	ConvBlock dsBlock{ nullptr };
	nn::ReLU relu{ nullptr };

	ResidualBlockImpl(ConvBlockParams &p, int64_t firstStride, ConvBlock downsample = nullptr);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ResidualBlock);

#endif