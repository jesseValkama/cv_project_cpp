#ifndef LENET_H
#define LENET_H

#include <cassert>
#include <optional>

#include <torch/torch.h>

#include "common.h"
#include "../settings.h"

namespace nn = torch::nn;
namespace ti = torch::indexing;

struct LeNetImpl : torch::nn::Module
{
	/*
	* The model comes from this post 
	* https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python
	* the model dynamically calculates the size for the fully connected layer
	* 
	* Attributes:
	*	cb1: settings for the 1st convblock
	*	cb2: settings for the 2nd convblock
	*	mp1: settings for the 1st maxpool
	*	mp2: settings for the 2nd maxpool
	*	ptrs: ptrs to the blocks
	*	cache: bool to cache a feature map for visualisation
	*	fm: the cached feature map
	*/

	ConvBlockParams cb1 = { 1, 6, 5, 1, 0, 6 };
	ConvBlockParams cb2 = { 6, 16, 5, 1, 0, 16 };
	MaxPoolParams mp1 = { 2, 2 };
	MaxPoolParams mp2 = { 2, 2 };

	ConvBlock conv1{ nullptr };
	ConvBlock conv2{ nullptr };
	nn::MaxPool2d mPool1{ nullptr };
	nn::MaxPool2d mPool2{ nullptr };
	nn::Linear fc1{ nullptr };
	nn::Linear fc2{ nullptr };
	nn::Linear fc3{ nullptr };
	nn::ReLU relu1{ nullptr };
	nn::ReLU relu2{ nullptr };

	bool cache = false;
	torch::Tensor fm;

	LeNetImpl(int nc, int imgsz, bool fmvis = false);
	torch::Tensor forward(torch::Tensor x);
	std::optional<torch::Tensor> get_fm(int fmi = -1);
	/*
	* Getter function for the cached featuremap
	* returns either all feature maps or a specific one
	* 
	* Args:
	*	fmi: index for the feature map, use -1 for all feature maps
	* 
	* Returns:
	*	feature map: success
	*	nullopt: failed (logged to terminal)
	*/
};

TORCH_MODULE(LeNet);

#endif