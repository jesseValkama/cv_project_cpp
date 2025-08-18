#include "settings.h"

#include "functions/train_lenet.h"

int main(void)
{
	Settings opts;
	int ret = 0;
	ret = lenet_loop(opts);
	if (ret != 0)
	{
		return ret;
	}

	return 0;
}

