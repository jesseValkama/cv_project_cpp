#ifndef DATABASE_H
#define DATABASE_H

#include <map>
#include <optional>
#include <sqlite3.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "database/param_info.h"
#include "database/database_map.h"

/*
* acknowledgements:
* https://medium.com/@duthujoe/sqlite-c-a-beginners-guide-243beb3fb3de
*/

using TableConstructor = std::unordered_map<std::string, ParamInfo>;
using DatabaseConstructor = std::unordered_map<std::string, TableConstructor>;

namespace db
{

DatabaseConstructor get_database_constructor(const char *name, const char *model, const char *dataset, const char *trainTime, const size_t epochs, const char *weights, 
							const double recall, const double precision, const double accuracy, const char *optimiser);
/*
* **NEEDS CHANGE** if you add change the database tables e.g., if you want to add more columns
*/

int open(const char *filename, sqlite3 **database, const bool verbose = true);
/*
* Function to open the connection to the database
* Args:
*	filename: the absolute path to the database file, if doesnt exist, creates one
*	database: the ptr to the ptr of the database
*	verbose: whether to print to terminal
* Returns:
*	int: status (https://sqlite.org/rescode.html#ok) 
*/

int close(sqlite3 *database, const bool verbose = true);
/*
* Funcition to close the connection to the database
* Args:
*	database: the ptr to the database
*	verbose: whether to print to terminal
* Returns:
*	int: status (https://sqlite.org/rescode.html#ok)
*	bear in mind, it does not retry to close the database
*/

int exec(sqlite3 *database, QueryTypes queryType, const DatabaseMap &databaseMap, const bool verbose = true);
/*
* Genergal function to execute INSERT and CREATE
*/

} // end of db

#endif // DATABASE_H