#include "settings.h"

#include <unordered_map>
#include <string>

#include "functions/handle_args.h"
#include "functions/inference.h"
#include "functions/train_lenet.h"

int main(int argc, char *argv[])
{
	/*
	* todo:
	*
	* add inf
	* add fm vis
	* per class metrics?
	* check vectors (optimisation)
	* check file reading (unsafe)
	* check type casts in funcs/common and inf (f auto)
	* add docstrings
	* add readme
	*/
	std::unordered_map<std::string, bool> args = handle_args(argc, argv);

	Settings opts;
	int ret = 0;
	if (args["train"] || args["test"])
	{
		ret = lenet_loop(opts);
		if (ret != 0) { return ret; }
	}
	
	if (args["inference"])
	{
		ret = run_inference(opts);
		if (ret != 0) { return ret; }
	}
	return 0;
}

