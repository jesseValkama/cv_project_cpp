#ifndef EARLYSTOP_H 
#define EARLYSTOP_H

#include <optional>
#include <stdint.h>
#include <unordered_map>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"

namespace nn = torch::nn;

bool early_stopping(size_t mem, size_t wait, size_t epoch, bool imp);
/*
	* Returns condition whether to stop training
	* Logic: if n amout of validations have passed in a row
	*		 and the model hasn't improved in any of them
	*		 return true, otherwise false
	*
	* Args:
	*	mem: the amount of validations without improvements in a row before early stopping
	*	wait: the amount of epochs to wait before activating activating function (at the end there will be 0 failed improvements stored)
	*	epoch: the current epoch
	*	imp: whether the model improved last epoch
	* 
	* Returns:
	*	bool: whether to stop training or not
*/

std::pair<Info, Info> split_train_val_info(Info& trainValInfo, double trainProb);

// i think typedefs are justified, since the types are long and using auto would be even worse
typedef std::vector<std::unordered_map<std::string, uint32_t>> AllCm;
typedef std::unordered_map<std::string, uint32_t> ClsCm;
typedef std::unordered_map<std::string, float> AvgMetrics;

struct MetricsContainer
{
	/*
	* Container to store the metrics, needs functions such as create_mc and calc_mc
	* 
	* Attributes:
	*	cm: confusion matrix (n of tp, fp, fn) and not the actual table
	*	talbe: the actual table for confusion matrix
	*	metrics: the container for recall, precision and accuracy
	*/
	AllCm cm;
	torch::Tensor table;
	AvgMetrics metrics;

	MetricsContainer(AllCm cm, torch::Tensor table) 
		: cm(cm), table(table) {};

	void add(int64_t i, std::string term, std::pair<int64_t, int64_t> &p);
	/*
	* Adds fp, tp or fn for a class for the cm
	* Constructs the actual confusion matrix table
	* 
	* Args:
	*	i: the call to add fp, tp or fn to (for cm)
	*	term: the term to add (fp, tp or fn for cm)
	*	p: the pair of predicted and ground truth classes for constructing table
	*/
	
	void calc_metrics(uint32_t nCls);
	/*
	* Method to calculate the metrics
	* Metrics are not returned, they are constructed to the attribute metrics
	* use print_metric() to print them
	* 
	* Args:
	*	nCls: the number of classes
	*/

	AllCm get_cm();
	/*
	* Method to return the tp, fp and fn
	* 
	* Returns:
	*	cm: tp, fp, fn
	*/

	void print_cm();
	/*
	* Method to print the cm table
	*/

	void print_metrics();
	/*
	* Method to print out metrics recall, precision and accuracy
	*/
};

MetricsContainer create_mc(uint32_t nCls);
/*
* Creates the metrics container
* 
* Args:
*	nCls: number of classes (metrics stored class-wise)
* 
* Returns:
*	metrics container: see the class for more info
*/

void calc_cm(torch::Tensor &labels, torch::Tensor &logits, MetricsContainer &mc);
/*
* Constructs the confusion matrix table and stats (tp, fp, fn)
* 
* Args: 
*	labels: ground truth
*	logits: output of the model before softmax
*	mc: metrics container class
*/

#endif
