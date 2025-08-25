#ifndef HANDLEARGS_H
#define HANDLEARGS_H

#include <optional>
#include <string>
#include <unordered_map>

std::optional<std::unordered_map<std::string, bool>> handle_args(int argc, char* argv[]);
/*
* Function to handle command-line arguments
* 
* Args:
*	argc: n of args
*	argv: array for args
* 
* Returns:
*	hashmap: successful
*	nullopt: failed (logged to terminal)
*/

int handle_flags(std::unordered_map<std::string, bool> &requiredArgs, std::string &key);
/*
* Function to validate input flags
* 
* Args:
*	requiredArgs: arguments required to pass validation
*	key: the argument to be validated
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int handle_values(std::unordered_map<std::string, bool> &requiredArgs, std::string &key, std::string &v);
/*
* Function to validate input values and changes the requiredArgs based on the value
* 
* Args:
*	requiredArgs: all required arguments
*	key: the argument used to change requiredArgs
*	v: the value to be validated and used to potentially change the argument
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

#endif
