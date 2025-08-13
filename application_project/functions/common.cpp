#include "common.h"

#include <iostream>


bool early_stopping(int mem, bool imp)
{

	/*
	* Automatically stops training if mem 
	* amount of validations have passed and
	* the model hasn't improved in any of them
	* 
	* Args:
	* mem, the amount of validations stored to 
	* imp, whether the model improved last epoch
	*/

	static int dec = 0;
	if (!imp)
	{
		dec++;
	}

	if (dec == mem)
	{
		std::cout << "Early stopping stopped the training" << std::endl;
		return true;
	}

	return false;
}

void calc_metrics()
{
	
}
