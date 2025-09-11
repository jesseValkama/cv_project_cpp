#ifndef LOADMODEL_H
#define LOADMODEL_H

#include <torch/torch.h>

#include <optional>
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>

#include "../settings.h"
#include "lenet.h"
#include "resnet.h"

enum ModelTypes
{
	LeNetType = 0,
	ResNetType = 1
};
typedef std::variant<LeNet, ResNet> Model;

class ModelWrapper
{
	private:
		std::optional<Model> model;
		MnistOpts mnistOpts;

		void create_model(int16_t modelType, MnistOpts mnistOpts, bool fmvis = false);
		/*
		* Method for creating a model, called internally by constructor method
		*/

	public:

		ModelWrapper(int16_t modelType, MnistOpts mnistOpts, bool fmvis = false);
		/*
		* Construction method, automatically creates the model with create_model
		*/
	
		void save_model(std::string path);
		/*
		* Method for saving the model
		*/

		void load_model(std::string path);
		/*
		* Method for loading weights for a model
		*/

		void train();

		void eval();

		void to(torch::Device dev);

		std::vector<torch::Tensor> parameters(bool recurse = true);

		torch::Tensor forward(torch::Tensor);
		/*
		* Method for performing the forward pass
		* 
		* Abort:
		*	if the model wasn't created correctly before
		*/

		std::optional<torch::Tensor> get_fm(int fmi);
		/*
		* Method for getting the feature map
		* 
		* Returns:
		*	torch::Tensor: the feature map with gradients
		*	nullopt: if tried to call with fmvis = false
		*/
};

#endif