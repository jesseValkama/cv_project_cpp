#ifndef LENET_INF
#define LENET_INF

#include "../settings.h"

int run_inference(Settings &opts);

int lenet_inference(torch::Tensor imgs, Settings &opts);

#endif