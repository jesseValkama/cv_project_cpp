#ifndef CLASSMETRICS_H
#define CLASSMETRICS_H

#include <torch/torch.h>

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

// i think typedefs are justified, since the types are long and using auto would be even worse
typedef std::vector<std::unordered_map<std::string, uint32_t>> AllCm;
typedef std::unordered_map<std::string, uint32_t> ClsCm;
typedef std::unordered_map<std::string, float> AvgMetrics;

class MetricsContainer
{
	/*
	* Container to store the metrics, needs functions such as create_mc and calc_mc
	* 
	* Attributes:
	*	cm: confusion matrix (n of tp, fp, fn) and not the actual table
	*	talbe: the actual table for confusion matrix
	*	metrics: the container for recall, precision and accuracy
	*/
private:
	AllCm cm;
	torch::Tensor table;
	std::vector<float> recalls;
	std::vector<float> precisions;
	std::vector<float> accuracies;
	AvgMetrics metrics;
	uint32_t nCls;
	void print_avg_metrics();
	void print_class_metrics(int idx);

public:
	MetricsContainer(uint32_t nCls);
	/*
	* Constructor method, inits the cm and table
	*/

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
	
	void calc_metrics();
	/*
	* Method to calculate the metrics
	* Metrics are not returned, they are constructed to the attribute metrics
	* use print_metric() to print them
	* 
	* Args:
	*	nCls: the number of classes
	*/

	AllCm get_cm() const;
	/*
	* Method to return the tp, fp and fn
	* 
	* Returns:
	*	cm: tp, fp, fn
	*/

	void print_cm() const;
	/*
	* Method to print the cm table
	*/

	void print_metrics(int idx = -1);
	/*
	* Method to print out metrics: recall, precision and accuracy
	* 
	* Args:
	*	idx: which class to print (-2 average metrics, -1 all per class metrics, n class idx)
	*/
};

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
