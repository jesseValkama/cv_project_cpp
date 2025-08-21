#ifndef LENET_H
#define LENET_H

#include <cassert>
#include <memory>

#include <torch/torch.h>

#include "common.h"
#include "../settings.h"

namespace nn = torch::nn;

struct LeNetImpl : torch::nn::Module
{
	/*
	* The model comes from this post 
	* https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python
	* This model does not process image sizes other than 32x32, as the original LeNet did
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

	LeNetImpl(int nc, int imgsz);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(LeNet);

#endif