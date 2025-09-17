#include "common.h"

#include <stdint.h>
#include <iostream>

bool early_stopping(size_t mem, size_t wait, size_t epoch, bool imp)
{
	static size_t dec = 0;
	if (epoch < wait)
	{
		return false;
	}
	if (!imp)
	{
		std::cout << "The model did not improve" << std::endl;
		dec++;
	}
	else
	{
		dec = 0;
	}
	if (dec >= mem)
	{
		std::cout << "Early stopping stopped the training" << std::endl;
		return true;
	}
	return false;
}
