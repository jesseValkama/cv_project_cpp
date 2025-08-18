#ifndef EARLYSTOP_H 
#define EARLYSTOP_H

#include <stdint.h>
#include <unordered_map>
#include <string>

#include <torch/torch.h>

namespace nn = torch::nn;

bool early_stopping(int mem, bool imp);

std::unordered_map<std::string, uint32_t> calc_cm(torch::Tensor labels, torch::Tensor logits, float threshold);

std::unordered_map<std::string, float> calc_metrics(std::unordered_map<std::string, uint32_t> cm);

#endif
