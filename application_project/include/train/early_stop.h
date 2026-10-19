#ifndef EARLYSTOP_H 
#define EARLYSTOP_H

#include <cstddef>

bool early_stopping(const size_t mem, const size_t wait, const size_t epoch, const bool imp);
/*
	* Returns condition whether to stop training
	* Logic: if n amout of validations have passed in a row
	*		 and the model hasn't improved in any of them
	*		 return true, otherwise false
	*
	* Args:
	*	mem: the amount of validations without improvements in a row before early stopping
	*	wait: the amount of epochs to wait before activating activating function, does still count the fails in a row
	*	epoch: the current epoch
	*	imp: whether the model improved last epoch
	* 
	* Returns:
	*	bool: whether to stop training or not
*/

#endif // EARLYSTOP_H
