#include "model_wrapper.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../settings.h"
#include "lenet.h"
#include "resnet.h"

namespace fs = std::filesystem;

ModelWrapper::ModelWrapper(ModelTypes modelType, DatasetOpts mnistOpts, bool fmvis)
	: mnistOpts(mnistOpts)
{
	create_model(modelType, mnistOpts, fmvis);
}

void ModelWrapper::create_model(ModelTypes modelType, DatasetOpts mnistOpts, bool fmvis)
{
	switch (modelType)
	{
		case ModelTypes::LeNetType:
			model = LeNet(mnistOpts.imgresz, mnistOpts.numOfClasses, mnistOpts.numOfChannels, fmvis);
			name = "LeNet";
			break;

		case ModelTypes::ResNetType:
			model = ResNet(mnistOpts.imgresz, mnistOpts.numOfClasses, mnistOpts.numOfChannels, fmvis);
			name = "ResNet";
			break;

		default:
			std::cout << "Unrecognised model type" << std::endl;
			std::abort();
	}
}

void ModelWrapper::save_weights(std::string path)
{
	if (!model.has_value()) { std::abort(); }
	std::optional<std::string> fPath = format_path(mnistOpts.savepath + "/" + path, mnistOpts);
	if (!fPath.has_value()) { std::abort(); }
	std::visit([&](auto &m) { torch::save(m, fPath.value()); }, model.value());
}

void ModelWrapper::load_weights(std::string path)
{
	if (!model.has_value()) { std::abort(); }
	std::optional<std::string> fPath = format_path(mnistOpts.savepath + "/" + path, mnistOpts);
	if (!fPath.has_value()) { std::abort(); }
	std::visit([&](auto &m) { torch::load(m, fPath.value()); }, model.value());
}

void ModelWrapper::print_layers()
{
	if (!model.has_value()) { std::abort; }
	std::visit([&](auto &m)
		{for (const auto &layer : m->named_modules())
	{
		std::cout << layer.key() << " : " << *layer.value() << std::endl;
	}}, (*model));
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

void ModelWrapper::to(torch::Device dev, bool non_blocking)
{
	if (!model.has_value()) { std::abort(); }
	std::visit([&](auto &m) { m->to(dev, non_blocking); }, model.value());
}

std::vector<torch::Tensor> ModelWrapper::parameters(bool recurse)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->parameters(recurse); }, model.value());
}

std::string ModelWrapper::get_name()
{
	return name;
}

torch::Tensor ModelWrapper::forward(torch::Tensor x)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->forward(x); }, model.value());
}

std::optional<torch::Tensor> ModelWrapper::get_fm(int16_t fmi)
{
	if (!model.has_value()) { std::abort(); }
	return std::visit([&](auto &m) { return m->get_fm(fmi); }, model.value());
}

std::optional<std::string> format_path(std::string path, DatasetOpts &mnistOpts)
{
	//todo: error handling
	std::regex rx(R"(^.+\.pth$)");
	if (std::regex_match(path, rx))
	{
		return path;
	}
	
	return path + ".pth";
}
