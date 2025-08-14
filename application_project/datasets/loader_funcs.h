#ifndef LOADMNIST_H
#define LOADMNIST_H

#include <torch/torch.h>

#include <stdint.h>
#include <string>
#include <tuple>
#include <vector>

typedef std::vector<std::pair<std::string, int64_t>> Data;
typedef torch::data::Example<> Example;

int load_mnist(std::string path, int64_t label, int imgSize, Example &o);
/*
* This function is used to load mnist dataset
* return statements indicate conditions
*/

int load_mnist_info(std::string fname, uint8_t trainProb, std::tuple<Data, Data, Data> &o);
/*
* This function is used to load mnist dataset info
* return statements indicate conditions
*/

#endif