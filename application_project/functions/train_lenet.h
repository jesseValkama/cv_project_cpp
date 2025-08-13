#ifndef TRAINLOOP_H
#define TRAINLOOP_H

#include <torch/torch.h>

#include <string>

#include "../models/lenet.h"
#include "../settings.h"

int lenet_loop(Settings &opts);
/*
	* This is the main training loop
	* return statements indicate conditions
*/

template<typename Dataloader>
int lenet_train(LeNet &model, Dataloader &trainloader, Dataloader &valloader, 
	torch::optim::Optimizer &optimiser, nn::CrossEntropyLoss &lossFn, Settings &opts);
/*
	* The function to train LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

template<typename Dataloader>
int lenet_val(LeNet &model, Dataloader &valloader, float &bestValLoss, nn::CrossEntropyLoss &lossFn, bool &imp);
/*
	* The function to validate LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

template<typename Dataloader>
int lenet_test(LeNet &model, Dataloader &testloader, nn::CrossEntropyLoss &lossFn);
/*
	* The function to test LeNet
	* 
	* Return statements indicate conditions
	* 
	* Args:
*/

#endif