#include "functions/common.h"

#include <stdint.h>
#include <iostream>

bool early_stopping(size_t mem, size_t wait, size_t epoch, bool imp)
{
	static size_t dec = 0;
	static bool active = false;
	if (!imp)
	{
		dec++;
		if (active)
		{
			std::cout << "The model did not improve " << "(" << dec << "/" << mem << ")" << std::endl;
		}
	}
	else
	{
		dec = 0;
	}

	if (epoch < wait)
	{
		return false;
	}
	else if (epoch == wait)
	{
		std::cout << "Early stopping is now active" << "(" << dec << "/" << mem << ")" << std::endl;
		active = true;
	}

	if (dec >= mem)
	{
		std::cout << "Early stopping stopped the training" << std::endl;
		return true;
	}
	return false;
}
