#include "common.h"

#include <stdint.h>
#include <iostream>
#include <unordered_map>
#include <vector>

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

std::vector<std::unordered_map<std::string, uint32_t>> calc_cm(torch::Tensor labels, torch::Tensor logits)
{
	/*
	* never done this before so i had to study it: 
	* i was only familiar with confusion matrices for object detection
	* src: https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/ 
	* 
	* constructing the cm
	*
	* repeat for each item
	* check lbl and pred
	*   1. go to the row that corresponds to the label
	*	2. add 1 to the prediction 
	*	   (if the prediction is wrong, it will automatically be both an fp and fn,
	*		because if it is a fp for one class, it will be a false negative for another)
	* 
	* therefore:
	*    if (lbl == pred)
	*		tp++
	*    else
	*		fp++
	*	    fn++
	* 
	*	     preds
	* 
	* l		s   v   t 
	* a  s [16, 0,  0] 0 + 0 for fn
	* b  v [0,  17, 1] 0 + 1 for fn
	* e  t [0,  0, 11] 0 + 0 for fn
	* l		0   0   0
	*		+   +   +
	*       0   0   1
	*       fp  fp  fp
	* 
	*/ 
	
	torch::Tensor preds = torch::softmax(logits, 1);
	std::vector<std::unordered_map<std::string, uint32_t>> cmstats;
	auto nc = labels.sizes();
	for (int i = 0; i < nc[0]; ++i)
	{
		cmstats.push_back({ {"tp", 0}, {"fp", 0}, {"fn", 0} });
	}

	auto n = preds.sizes(); // TODO: typecast to int
	for (int i = 0; i < n[0]; ++i)
	{
		torch::Tensor xi = torch::argmax(preds[i]);
		torch::Tensor prob = preds[i][xi];
		torch::Tensor label = labels[i];

		xi = xi.to(torch::kCPU);
		label = label.to(torch::kCPU);
		prob = prob.to(torch::kCPU);
		int64_t p = xi.item<int64_t>();
		int64_t l = label.item<int64_t>();
		
		// TODO: construct the cm

		if (p == l)
		{
			cmstats[l]["tp"] += 1;
		}
		else
		{
			cmstats[l]["fp"] += 1;
			cmstats[p]["fn"] += 1;
		}
	}
	
	return cmstats;
}

std::unordered_map<std::string, float> calc_metrics(std::unordered_map<std::string, uint32_t> cm)
{
	std::unordered_map<std::string, float> metrics =
	{ {"recall",0.0},{"precision",0.0},{"accuracy",0.0} };
	return metrics;
}
