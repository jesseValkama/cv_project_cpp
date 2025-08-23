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
	* check vectors (optimisation)
	* check strings (unsafe?)
	* check file reading (unsafe)
	* check type casts in funcs/common and inf (f auto)
	* add docstrings
	* add readme
	* todo errors
	*/
	std::unordered_map<std::string, bool> args = handle_args(argc, argv);
	Settings opts;
	int ret = 0;

	if (args["train"] || args["test"])
	{
		ret = lenet_loop(opts, args["train"], args["test"]);
		if (ret != 0) { return ret; }
	}
	if (args["inference"])
	{
		run_inference(opts);
	}
	return 0;
}

