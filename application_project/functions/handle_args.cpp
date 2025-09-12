#include "handle_args.h"

#include <cstring>
#include <iostream>
#include <optional>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unordered_map>

std::optional<std::unordered_map<std::string, int16_t>> handle_args(int argc, char *argv[])
{
	std::unordered_map<std::string, int16_t> requiredArgs = { {"train", 0}, {"test", 0}, {"inference", 0}, {"xai", 0}, {"model", 0} };
	if (requiredArgs.size() * 2 != argc - 1)
	{
		std::cout << "Incorrect n of command-line arguments" << std::endl;
		return std::nullopt;
	}
	int ret = 0;
	std::string key = "";
	std::string v = "";
	for (int i = 1; i < argc; ++i)
	{
		
		char* tmp = argv[i];
		int n = strlen(tmp);
		if (i % 2 == 1)
		{
			key.assign(tmp, n);
			ret = handle_flags(requiredArgs, key);
			if (ret != 0) { return std::nullopt; }
		}
		else
		{
			v.assign(tmp, n);
			ret = handle_values(requiredArgs, key, v);
			if (ret != 0) { return std::nullopt; }
		}
	}
	return requiredArgs;
}

int handle_flags(std::unordered_map<std::string, int16_t>& requiredArgs, std::string &key)
{
	if (!(requiredArgs.find(key) != requiredArgs.end()))
	{
		std::cout << "unexpected command-line argument" << std::endl;
		return 1;
	}
	return 0;
}

int handle_values(std::unordered_map<std::string, int16_t> &requiredArgs, std::string &key, std::string &v)
{
	std::string::const_iterator it = v.begin();
	if (it != v.end() && *it == '-') { ++it; }
	while (it != v.end() && std::isdigit(*it)) { ++it; }
	if (!v.empty() && it == v.end())
	{
		requiredArgs[key] = atoi(v.c_str());
		return 0;
	}
	std::cout << "Unexpected input " << v << " for " << key << std::endl;
	return 1;
}