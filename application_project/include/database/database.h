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

#include "settings.h"
#include "database/database_helpers/param_info.h"
#include "database/database_helpers/database_map.h"

/*
* acknowledgements:
* https://medium.com/@duthujoe/sqlite-c-a-beginners-guide-243beb3fb3de
*/

using TableConstructor = std::unordered_map<std::string, ParamInfo>;
using DatabaseConstructor = std::unordered_map<std::string, TableConstructor>;

std::string make_attribute_list_template(const TableConstructor &tableConstructor, const bool usePlaceholder = false, const bool includeTypes = false);
/*
* Reference doesnt work here because of the for each loop, so it needs to make a copy
*/

int bind_values(sqlite3_stmt *&statement, const ParamInfo &paramInfo, int &idx);
/*
*/

struct QueryTemplateContainer
{
	/*
	* Initiates the queries to nullptr and might return nullptr queries, you need to check them manually
	*/
	std::map<std::string, std::string> queryTemplates;

	QueryTemplateContainer(const DatabaseMap &databaseMap);
	
	void make_create_templates(const DatabaseMap &databaseMap);

	void make_insert_templates(const DatabaseMap &databaseMap);
	
};

namespace db
{

DatabaseConstructor get_database_constructor(const char *name, const char *model, const char *trainTime, const size_t epochs, const char *weights, 
							const double recall, const double precision, const double accuracy, const char *optimiser, const Settings &settings);
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

int create_experiments(sqlite3 *database, const bool verbose = true);
/*
* not finished
* Function assumes that the table experiments **DOES NOT** exist
* safe as long as the queries are not messed around with
*/

int insert_experiments(sqlite3 *database, const DatabaseMap &databaseMap, const bool verbose = true);
/*
* **NEEDS CHANGE** if you add change the database tables e.g., if you want to add more columns
*/

} // end of db

// the rest of them are not necessary

int unsafe_read(sqlite3 *database, const char *query, const bool verbose = true);
/*
* UNSAFE function for reading as it expects the query to be passed as the parameter and the funcition
* **DOES NOT** include any sanitisation, not a recommended function as it needs a rewrite
* Args:
*	database: the ptr to the database, passing nullptr returns 1
*	query: the query for READING
*	verbose: whether it prints errors e.g., passing nullptr as database
* Returns:
*	status: 1 or (https://sqlite.org/rescode.html#ok)
*	bear in mind, sqlite also includes 1 TODO: fix
*/

const char *get_query(const Queries query);
/*
* not finished
* Returns:
*	nullptr: if invalid query option
*/

int exec(sqlite3 *database, const Queries query, const char *param, const bool verbose = true);
/*
* not finished
*/

#endif // DATABASE_H