#include "settings.h"

#include "datasets/mnist.h"

#include "functions/common.h"
#include "functions/train_lenet.h"

#include "datasets/tmp.h"

int main(void)
{
	
	Settings opts;
	int tmp = 0;
	tmp = vis_mnist();
	if (tmp != 0)
	{
		return tmp;
	}
	
	int ret = 0;
	ret = lenet_loop(opts);
	if (ret != 0)
	{
		return ret;
	}

	return 0;
}

