#ifndef EARLYSTOP_H 
#define EARLYSTOP_H

#include <stdint.h>
#include <unordered_map>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"

namespace nn = torch::nn;

bool early_stopping(int mem, bool imp);
std::pair<Info, Info> split_train_val_info(Info& trainValInfo, double trainProb);

// i think typedefs are justified, since the types are long and using auto would be even worse
typedef std::vector<std::unordered_map<std::string, uint32_t>> AllCm;
typedef std::unordered_map<std::string, uint32_t> ClsCm;
typedef std::unordered_map<std::string, float> AvgMetrics;

struct MetricsContainer
{
	AllCm cm;
	torch::Tensor table;
	AvgMetrics metrics;

	MetricsContainer(AllCm cm, torch::Tensor table) 
		: cm(cm), table(table) {};

	void add(int64_t i, std::string term, std::pair<int64_t, int64_t> &p);
	
	void calc_metrics(uint32_t nc);

	AllCm get_cm();

	void print_cm();

	void print_metrics();

};

MetricsContainer create_mc(uint32_t nc);

void calc_cm(torch::Tensor labels, torch::Tensor logits, MetricsContainer &mc);

#endif
