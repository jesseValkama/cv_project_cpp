#include "common.h"

#include <stdint.h>
#include <iostream>
#include <unordered_map>

#include <torch/torch.h>


bool early_stopping(int mem, bool imp)
{

	/*
	* Automatically stops training if mem 
	* amount of validations have passed and
	* the model hasn't improved in any of them
	* 
	* Args:
	* mem, the amount of validations stored to 
	* imp, whether the model improved last epoch
	*/

	static int dec = 0;
	if (!imp)
	{
		dec++;
	}

	if (dec == mem)
	{
		std::cout << "Early stopping stopped the training" << std::endl;
		return true;
	}

	return false;
}

std::unordered_map<std::string, uint32_t> calc_cm(torch::Tensor labels, torch::Tensor logits, float threshold)
{
	/*
	* PSEUDO CODE
	* tp = correct && prob > threshold
	* fp = incorrect && prob > threshold
	* fn = incorrect || correct && prob < threshold
	*/

	torch::Tensor preds = torch::softmax(logits, 1);
	std::unordered_map<std::string, uint32_t> cm =
	{ {"tp", 0}, {"fp", 0}, {"fn", 0} };
	
	auto size = preds.sizes();
	for (int i = 0; i < size[0]; ++i)
	{
		torch::Tensor xi = torch::argmax(preds[i]);
		torch::Tensor prob = preds[i][xi];
		torch::Tensor label = labels[i];

		xi = xi.to(torch::kCPU);
		label = label.to(torch::kCPU);
		prob = prob.to(torch::kCPU);

		if (prob.item<float>() < threshold)
		{
			cm["fn"] += 1;
		}
		else if (xi.item<int64_t>() == label.item<int64_t>())
		{
			cm["tp"] += 1;
		}
		else
		{
			cm["fp"] += 1;
		}
	}

	std::cout << "tp: " << cm["tp"] << std::endl;
	std::cout << "fp: " << cm["fp"] << std::endl;
	std::cout << "fn: " << cm["fn"] << std::endl;

	return cm;
}

std::unordered_map<std::string, float> calc_metrics(std::unordered_map<std::string, uint32_t> cm)
{
	std::unordered_map<std::string, float> metrics =
	{ {"recall",0.0},{"precision",0.0},{"accuracy",0.0} };
	return metrics;
}
