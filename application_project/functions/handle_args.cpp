#include "handle_args.h"

#include <iostream>
#include <optional>
#include <string>
#include <string.h>
#include <unordered_map>

std::optional<std::unordered_map<std::string, bool>> handle_args(int argc, char *argv[])
{
	std::unordered_map<std::string, bool> requiredArgs = { {"train", false}, {"test", false}, {"inference", false} };
	if (requiredArgs.size() * 2 != argc - 1)
	{
		std::cout << "Incorrect n of command-line arguments" << std::endl;
		return std::nullopt;
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

int handle_flags(std::unordered_map<std::string, bool>& requiredArgs, std::string &key)
{
	if (!(requiredArgs.find(key) != requiredArgs.end()))
	{
		std::cout << "unexpected command-line argument" << std::endl;
		return 1;
	}
	return 0;
}

 int handle_values(std::unordered_map<std::string, bool> &requiredArgs, std::string &key, std::string &v)
{
	if (v != "0" && v != "1")
	{
		std::cout << "expected value to be 0 or 1" << std::endl;
		return 1;
	}
	bool m = (v == "1") ? true : false;
	requiredArgs[key] = m;
	return 0;
}