#include "optims/scheds/warmup.h"

#include <torch/optim.h>

#include <vector>

namespace optims 
{
namespace sched
{
	warmupLR::warmupLR(torch::optim::Optimizer &optimiser, double lr, int iters_tot, int iters_int)
		: torch::optim::LRScheduler(optimiser), lr(lr), iters_tot(iters_tot), iters_int(iters_int) {};

	std::vector<double> warmupLR::get_lrs()
	{
		if (iters_cur == iters_tot)
		{
			return get_current_lrs();
		}
		iters_cur += -1 * iters_int;
		std::vector lrs = get_current_lrs();
		
		// v is not used, since LRScheduler always sets the value to the last value
		// -> the formula would not work since the lr is increasing
		std::transform(
        lrs.begin(), lrs.end(), lrs.begin(), [this](const double& v) {
          return this->lr * this->iters_cur / this->iters_tot;
        });

		return lrs;
	}

} // end of scheds
} // end of optims