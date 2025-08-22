#ifndef HANDLEARGS_H
#define HANDLEARGS_H

#include <string>
#include <unordered_map>

std::unordered_map<std::string, bool> handle_args(int argc, char* argv[]);
void handle_flags(std::unordered_map<std::string, bool> &requiredArgs, std::string &key);
void handle_values(std::unordered_map<std::string, bool> &requiredArgs, std::string &key, std::string &v);

#endif
