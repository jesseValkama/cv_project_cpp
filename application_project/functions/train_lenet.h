#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <torch/torch.h>

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../models/lenet.h"
#include "../settings.h"

int lenet_loop(Settings &opts);
/*
	* This is the main training loop
	* return statements indicate conditions
*/

template<typename Dataloader>
int lenet_train(Dataloader &trainloader, Dataloader &valloader, Settings &opts);
/*
	* The function to train LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

template<typename Dataloader>
int lenet_val(LeNet &model, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp, Settings &opts);
/*
	* The function to validate LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

template<typename Dataloader>
int lenet_test(Dataloader &testloader, Settings &opts);
/*
	* The function to test LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

#endif