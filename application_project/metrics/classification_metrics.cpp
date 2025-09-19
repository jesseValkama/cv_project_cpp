#include "classification_metrics.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

MetricsContainer::MetricsContainer(uint32_t cls)
{
	nCls = cls;
	for (int i = 0; i < nCls; ++i)
	{
		cm.push_back({ {"tp", 0}, {"fp", 0}, {"fn", 0} });
	}
	table = torch::zeros({nCls, nCls});
}

void MetricsContainer::add(int64_t i, std::string term, std::pair<int64_t, int64_t> &p)
{
	cm[i][term] += 1;
	table[p.first][p.second] += 1;
}

AllCm MetricsContainer::get_cm() const
{
	return cm;
}

void MetricsContainer::print_cm() const
{
	std::cout << "The confusion matrix:\n" << table << std::endl;
}

void MetricsContainer::calc_metrics()
{
	float recall = 0.0, precision = 0.0, accuracy = 0.0;
	float currentRecall = 0.0, currentPrecision = 0.0, currentAccuracy = 0.0;
	for (int i = 0; i < nCls; ++i)
	{
		currentRecall = (float) cm[i]["tp"] / ((float) cm[i]["tp"] + (float) cm[i]["fn"]);
		recalls.push_back(currentRecall);
		recall += currentRecall;
		currentPrecision = (float) cm[i]["tp"] / ((float) cm[i]["tp"] + (float) cm[i]["fp"]);
		precisions.push_back(currentPrecision);
		precision += currentPrecision;
		currentAccuracy = (float) cm[i]["tp"] / ((float) cm[i]["tp"] + (float) cm[i]["fp"]); // each incorrect prediction adds both a fp and fn for different classes
		accuracies.push_back(currentAccuracy);
		accuracy += currentAccuracy;
	}
	recall /= (float) nCls;
	precision /= (float) nCls;
	accuracy /= (float) nCls;

	metrics = { {"recall", recall},{"precision", precision},{"accuracy", accuracy} };
}

void MetricsContainer::print_metrics(int idx)
{
	assert(idx >= -2);
	std::cout << std::fixed;
	std::cout << std::setprecision(2);
	
	std::cout << "\n";
	std::cout << "----------------------------------------------METRICS----------------------------------------------" << std::endl;
	std::cout << "\n";

	if (idx == -2) { print_avg_metrics(); }
	else { print_class_metrics(idx); }

	std::cout << "-------------------------------------------END OF METRICS-------------------------------------------" << std::endl;
	std::cout << "\n";

}

void MetricsContainer::print_avg_metrics()
{
	std::cout << "recall: " << metrics["recall"] << std::endl;
	std::cout << "precision: " << metrics["precision"] << std::endl;
	std::cout << "accuracy: " << metrics["accuracy"] << std::endl;
	std::cout << "\n";
}

void MetricsContainer::print_class_metrics(int idx)
{
	if (idx == -1)
	{
		int n = recalls.size();
		for (int i = 0; i < n; ++i)
		{
			std::cout << "recall for " << i << ": " << recalls[i] << std::endl;
			std::cout << "precision for " << i << ": " << precisions[i] << std::endl;
			std::cout << "accuracy for " << i << ": " << accuracies[i] << std::endl;
			std::cout << "\n";
		}
	}
	else
	{
		std::cout << "recall for " << idx <<":" << recalls[idx] << std::endl;
		std::cout << "precision for " << idx << ":" << precisions[idx] << std::endl;
		std::cout << "accuracy for " << idx << ":" << accuracies[idx] << std::endl;
		std::cout << "\n";
	}
}

void calc_cm(torch::Tensor &labels, torch::Tensor &logits, MetricsContainer &mc)
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
	
	int n = preds.size(0);
	for (int i = 0; i < n; ++i)
	{
		torch::Tensor xi = torch::argmax(preds[i]);
		torch::Tensor prob = preds[i][xi];
		torch::Tensor label = labels[i];

		xi = xi.to(torch::kCPU);
		label = label.to(torch::kCPU);
		prob = prob.to(torch::kCPU);
		int64_t p = xi.item<int64_t>();
		int64_t l = label.item<int64_t>();
		
		std::pair<int64_t, int64_t> pair = std::make_pair(l, p);
		if (p == l)
		{
			mc.add(l, "tp", pair);
		}
		else
		{
			mc.add(l, "fp", pair);
			mc.add(p, "fn", pair);
		}
	}
}
