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

typedef std::variant<LeNet, ResNet> Model;

class ModelWrapper
{
	private:
		std::optional<Model> model;
		DatasetOpts mnistOpts;
		std::string name = "none";

		void create_model(ModelTypes modelType, DatasetOpts mnistOpts, bool fmvis = false);
		/*
		* Method for creating a model, called internally by constructor method
		*/

	public:

		ModelWrapper(ModelTypes modelType, DatasetOpts mnistOpts, bool fmvis = false);
		/*
		* Construction method, automatically creates the model with create_model
		*/

		void save_weights(std::string path);
		void load_weights(std::string path);
		void print_layers() const;
		void train();
		void eval();
		void to(torch::Device dev, bool non_blocking = false);
		std::vector<torch::Tensor> parameters(bool recurse = true) const;
		std::string get_name() const;
		
		torch::Tensor forward(torch::Tensor);
		/*
		* Method for performing the forward pass
		* 
		* Abort:
		*	if the model wasn't created correctly before
		*/

		std::optional<torch::Tensor> get_fm(int16_t fmi);
		/*
		* Method for getting the feature map
		* 
		* Returns:
		*	torch::Tensor: the feature map with gradients
		*	nullopt: if tried to call with fmvis = false
		*/
};

std::optional<std::string> format_path(std::string path, DatasetOpts &mnistOpts);
/*
* Automatically adds the .pth extention if it is not present in the path
* TODO: error handling for paths
* 
* Args:
*	path: the actual path
*	mnistOpts: the options for the path
* 
* Returns:
*	std::string: the formatted path
*	nullopt: the path doesn't exist
*/

#endif