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

	nn::Conv2d conv1{ nullptr };
	nn::BatchNorm2d bn1{ nullptr };
	nn::ReLU a1{ nullptr };

	nn::Conv2d conv2{ nullptr };
	nn::BatchNorm2d bn2{ nullptr };
	nn::ReLU a2{ nullptr };

	//ConvBlock conv1{ nullptr };
	//ConvBlock conv2{ nullptr };
	nn::MaxPool2d mp1{ nullptr };
	nn::MaxPool2d mp2{ nullptr };
	nn::Linear fc1{ nullptr };
	nn::Linear fc2{ nullptr };
	nn::Linear fc3{ nullptr };
	nn::ReLU relu1{ nullptr };
	nn::ReLU relu2{ nullptr };

	LeNetImpl(int nc);
	torch::Tensor forward(torch::Tensor x);
	
	};

TORCH_MODULE(LeNet);

#endif