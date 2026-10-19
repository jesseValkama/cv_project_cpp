#include "settings/handle_args.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

int args::Parser::add_arguments(const std::vector<args::ArgInfo> &argsinfo)
{
    for (const auto &arginfo : argsinfo)
    {
        if (this->args.find(arginfo.argname) != this->args.end())
        {
            return ARGS_INVALID_ARG;
        }
        Arg arg = Arg();
        if (arginfo.conversionTable.has_value())
        {
            arg.conversionTable = arginfo.conversionTable;
        }
        this->args[arginfo.argname] = arg;
    }
    return ARGS_OK;
}

std::optional<Args> args::Parser::parse_args(int argc, char **argv)
{
	for (int i = 1; i < argc; i += 2)
    {
        std::string key = argv[i];
        std::string value = argv[i+1];
        std::optional<int16_t> v = this->convert_arg(this->args.at(key), value);
        if (!v.has_value())
        {
            return std::nullopt;
        }
        this->args[key].value = *v;
        this->args[key].meta = this->meta(value);
    }
    return this->args;
}

std::optional<int16_t> args::Parser::convert_arg(const Arg arg, const std::string &value)
{
    if (arg.conversionTable.has_value() && arg.conversionTable->find(value) != arg.conversionTable->end())
    {
        return arg.conversionTable->at(value);
    }
    std::string::const_iterator it = value.begin();
    if (it != value.end() && *it == '-') { ++it; }
    while (it != value.end() && std::isdigit(*it)) { ++it; }
    if (!value.empty() && it == value.end())
    {
        return atoi(value.c_str());
    }
    return std::nullopt;
}

std::optional<int16_t> args::Parser::meta(const std::string &value)
{
    if (value != "Idx")
    {
        return std::nullopt;
    }
    int16_t x;
    std::cout << "Which idx would you like to visualise?" << std::endl;
    if (std::cin >> x)
    {
        return x;
    }
    return std::nullopt;
}