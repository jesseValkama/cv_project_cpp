#include "application_project.h"

#include <optional>
#include <string>
#include <unordered_map>

#include "settings.h"
#include "functions/handle_args.h"
#include "functions/inference.h"
#include "functions/train_lenet.h"

int main(int argc, char *argv[])
{
	/*
	* todo:
	*
	* check type casts in funcs/common and inf (f auto)
	* complete readme
	* add unit tests
	*/
	std::optional<std::unordered_map<std::string, bool>> args = handle_args(argc, argv);
	if (!args.has_value()) { return 1; }
	Settings opts;
	int ret = 0;

	if (args.value()["train"] || args.value()["test"])
	{
		ret = lenet_loop(opts, args.value()["train"], args.value()["test"]);
		if (ret != 0) { return ret; }
	}
	if (args.value()["inference"])
	{
		ret = run_inference(opts);
		if (ret != 0) { return ret; }
	}
	return 0;
}

