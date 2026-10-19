#ifndef HANDLEARGS_H
#define HANDLEARGS_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "settings/settings.h"

enum ArgsConditions
{
    ARGS_OK = 0,
    ARGS_INVALID_ARG = 10,
};

struct Arg
{
    uint16_t value;
    std::optional<uint16_t> meta;
    std::optional<std::unordered_map<std::string, int16_t>> conversionTable;
};

using Args = std::unordered_map<std::string, Arg>;

namespace args
{

struct ArgInfo
{
    std::string argname;
    std::optional<std::unordered_map<std::string, int16_t>> conversionTable;

    ArgInfo(const std::string &argname, const std::optional<std::unordered_map<std::string, int16_t>> conversionTable = std::nullopt)
            : argname(argname), conversionTable(conversionTable) {};
};

class Parser
{
private:
    Args args;
public:
    Parser() {};

    int add_arguments(const std::vector<ArgInfo> &argsinfo);
    /*
    * Method to add cli arguments for the parser
    * Initialises the value to std::nullopt if the arg does not have a conversionMap defined,
    * otherwise initialises it according to the map
    * Args:
    *   argname: the name of the arg
    * Returns:
    *   ARGS_OK: if arg added successfully
    *   ARGS_INVALID_ARG: if arg already exists in the map
    */

    std::optional<Args> parse_args(int argc, char **argv);
    /*
    * Method to parse cli arguments
    * Args:
    *   args: the number of cli args (pass directly from main argc)
    *   argv: array of char * (pass directly from main argv)
    * Returns:
    *   nullopt: if input cli args do not match the defined ones
    *   Args: the unordered_map definning values for args
    */

private:

    std::optional<int16_t> convert_arg(const Arg arg, const std::string &value);
    /*
    * Helper method to convert args from std::string to int16_t
    * either maps directly based on the value or based on a conversionTable if that is provided
    * Args:
    *   arg: Arg object
    *   value: the value for arg
    * Returns:
    *   std::nullopt: the value cant be mapped to int16_t
    *   int16_t: the arg as int16_t
    */

    std::optional<int16_t> meta(const std::string &value);
    /*
    */

};

} // end of args

#endif // HANDLEARGS_H