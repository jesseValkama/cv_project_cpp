#ifndef DATAHELPER_H
#define DATAHELPER_H

#include <torch/torch.h>

#include <functional>
#include <tuple>
#include <variant>
#include <vector>

#include "../datasets/cifar10.h"
#include "../datasets/loader_funcs.h"
#include "../datasets/mnist.h"
#include "../settings.h"

namespace nn = torch::nn;

typedef torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<MnistDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>> Mnistds;
typedef torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<Cifar10Dataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>> Cifar10ds;

std::tuple<Info, Info, Info> load_dataset_info(DatasetTypes datasetType, DatasetOpts &datasetOpts, std::vector<int> &tidxs, std::vector<int> &vidxs, double trainRatio = 0.85); 

std::pair<Info, Info> split_train_val_info(Info& trainValInfo, double trainProb, int bs, std::vector<int> &tidxs, std::vector<int> &vidxs);

std::tuple<Mnistds, Mnistds, Mnistds> make_mnist_datasets(DatasetOpts &datasetOpts, const Info &trainInfo, const Info &valInfo, const Info &testInfo);

std::tuple<Cifar10ds, Cifar10ds, Cifar10ds> make_cifar10_datasets(DatasetOpts &datasetOpts, const Info &trainInfo, const Info &valInfo, const Info &testInfo, const std::vector<int> &tidxs, const std::vector<int> &vidxs);

#endif
