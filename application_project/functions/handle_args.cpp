#include "handle_args.h"

#include <stdexcept>
#include <string>
#include <string.h>
#include <unordered_map>

std::unordered_map<std::string, bool> handle_args(int argc, char *argv[])
{
	std::unordered_map<std::string, bool> requiredArgs = { {"train", false}, {"test", false}, {"inference", false} };
	if (requiredArgs.size() * 2 != argc - 1)
	{
		throw std::invalid_argument("Incorrect n of command-line arguments");
	}
	std::string key = "";
	std::string v = "";
	for (int i = 1; i < argc; ++i)
	{
		
		char* tmp = argv[i];
		int n = strlen(tmp);
		if (i % 2 == 1)
		{
			key.assign(tmp, n);
			handle_flags(requiredArgs, key);
		}
		else
		{
			v.assign(tmp, n);
			handle_values(requiredArgs, key, v);
		}
	}
	return requiredArgs;
}

void handle_flags(std::unordered_map<std::string, bool>& requiredArgs, std::string &key)
{
	if (!(requiredArgs.find(key) != requiredArgs.end()))
	{
		throw std::invalid_argument("Unexpected command-line argument");
	}
}

void handle_values(std::unordered_map<std::string, bool> &requiredArgs, std::string &key, std::string &v)
{
	if (v != "0" && v != "1")
	{
		throw std::invalid_argument("Expected value to be 0 or 1");
	}
	bool m = (v == "1") ? true : false;
	requiredArgs[key] = m;
}