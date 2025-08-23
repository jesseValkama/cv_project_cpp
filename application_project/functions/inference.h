#ifndef LENET_INF
#define LENET_INF

#include <string>
#include <vector>

#include "../settings.h"

void run_inference(Settings &opts);

void lenet_inference(std::vector<std::string> &fImgs, Settings &opts);

#endif