#include "model_wrapper.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../settings.h"
#include "lenet.h"
#include "resnet.h"

ModelWrapper::ModelWrapper(int16_t modelType, MnistOpts mnistOpts, bool fmvis)
	: mnistOpts(mnistOpts)
{
	create_model(modelType, mnistOpts, fmvis);
}

void ModelWrapper::create_model(int16_t modelType, MnistOpts mnistOpts, bool fmvis)
{
	ModelTypes type = static_cast<ModelTypes>(modelType);
	switch (type)
	{
		case ModelTypes::LeNetType:
			model = LeNet(mnistOpts.numOfClasses, mnistOpts.imgresz, fmvis);
			break;

		case ModelTypes::ResNetType:
			model = ResNet(mnistOpts.imgresz, mnistOpts.numOfClasses, 1);
			break;

		default:
			std::cout << "Unrecognised model type" << std::endl;
			std::abort();
	}
}

void ModelWrapper::save_model(std::string path)
{
	// todo error handling
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { torch::save(m, mnistOpts.savepath + "/" + path); }, model.value());
}

void ModelWrapper::load_model(std::string path)
{
	// todo error handling
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { torch::load(m, mnistOpts.savepath + "/" + path); }, model.value());
}

void ModelWrapper::train()
{
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { m->train(); }, model.value());
}

void ModelWrapper::eval()
{ 
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { m->eval(); }, model.value());
}

void ModelWrapper::to(torch::Device dev)
{
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { m->to(dev); }, model.value());
}

std::vector<torch::Tensor> ModelWrapper::parameters(bool recurse)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->parameters(recurse); }, model.value());
}

torch::Tensor ModelWrapper::forward(torch::Tensor x)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->forward(x); }, model.value());
}

std::optional<torch::Tensor> ModelWrapper::get_fm(int fmi)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->get_fm(fmi); }, model.value());
}
