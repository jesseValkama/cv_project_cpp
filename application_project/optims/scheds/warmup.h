#ifndef WARMUP_H
#define WARMUP_H

#include <torch/optim.h>

#include <vector>

namespace optims 
{ 
namespace sched
{
	/*
	* warmup scheduler for training
	* Uses the formula I_cur / I_total * lr
	* 
	* The scheduler StepLR was used as a reference
	* https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/optim/schedulers/step_lr.cpp
	*/

	class warmupLR : public torch::optim::LRScheduler
	{

	public:
		warmupLR(torch::optim::Optimizer &optimiser, double lr, int iters_tot, int iters_int = -1);

	private:
		double lr = 0.0;
		int iters_tot = 0;
		int iters_cur = 0;
		int iters_int = 0;
		std::vector<double> get_lrs() override;

	};

} // end of namespace scheds
} // end of namespace optims

#endif