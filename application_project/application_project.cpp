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
	* add idx as a command-line arg (remove form opts)
	* fix probabilites
	* add wait to early stopping
	* complete readme
	* add unit tests (does googletest even work?)
	*/
	
	// this is here, because the settings.h takes a while to build
	int XAI = 10;

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
		ret = run_inference(opts, XAI);
		if (ret != 0) { return ret; }
	}
	return 0;
}

