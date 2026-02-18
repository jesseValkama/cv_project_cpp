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

/*
* acknowledgements:
* https://medium.com/@duthujoe/sqlite-c-a-beginners-guide-243beb3fb3de
*/

enum SqliteTypes
{
	TYPE_INT = 1,
	TYPE_REAL = 2,
	TYPE_TEXT = 3,
	TYPE_BLOB = 4,
	TYPE_NULL = 5
};

enum Queries
{
	CREATE = 0,
	INSERT = 1,
	UPDATE = 2
};

struct ForeignKey
{
	using QueryVariant = std::variant<const char *, int, double>;

	std::string referenceTable = "";
	std::string referenceColumn = "";
	bool cascade = true;

	ForeignKey(const std::string referenceTable, const std::string referenceColumn, const bool cascade = true)
		: referenceTable(referenceTable), referenceColumn(referenceColumn), cascade(cascade) {};
};

struct ParamInfo
{
	using QueryVariant = std::variant<const char *, int, double>;

	QueryVariant value = 0;
	SqliteTypes type = TYPE_NULL;
	bool notNull = false;
	bool unique = false;
	bool autoIncrement = false;
	bool primaryKey = false;
	std::optional<ForeignKey> foreignKey = std::nullopt;

	ParamInfo(const QueryVariant value, const SqliteTypes type, const bool notNull = false, const bool unique = false, const bool autoIncrement = false, const bool primaryKey = false, const std::optional<ForeignKey> foreignKey = std::nullopt) 
		: value(value), type(type), notNull(notNull), unique(unique), autoIncrement(autoIncrement), primaryKey(primaryKey), foreignKey(foreignKey) {};
};

using TableConstructor = std::unordered_map<std::string, ParamInfo>;
using DatabaseConstructor = std::unordered_map<std::string, TableConstructor>;

std::string make_attribute_list_template(const TableConstructor &tableConstructor, const bool usePlaceholder = false, const bool includeTypes = false);
/*
* Reference doesnt work here because of the for each loop, so it needs to make a copy
*/

int bind_values(sqlite3_stmt *&statement, const ParamInfo &paramInfo, int &idx);
/*
*/

class DatabaseIterator
{
	/*
	* this class is heavilty inspired the tutorial:
	* https://www.youtube.com/watch?v=F9eDv-YIOQ0
	* 
	* might be useful:
	* https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp
	*/
	 DatabaseConstructor databaseConstructor;
	 std::vector<std::string> tableOrder;
	 int orderIdx = 0;
public:
	DatabaseIterator(DatabaseConstructor &databaseConstructor, std::vector<std::string> &tableOrder) : databaseConstructor(databaseConstructor), tableOrder(tableOrder) {};
	DatabaseIterator &operator++();
	DatabaseIterator &operator--();
	DatabaseIterator operator++(int);
	DatabaseIterator operator--(int);
	std::pair<std::string, TableConstructor> operator*();
	bool operator==(const DatabaseIterator &other) const;
	bool operator!=(const DatabaseIterator &other) const;
};

class DatabaseMap
{
	/*
	* acknowledgements:
	*	https://www.geeksforgeeks.org/python/introduction-to-graphs-in-python/
	*	https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
	*/
public:
	using Iterator = DatabaseIterator;
	
private:
	DatabaseConstructor databaseConstructor;
	std::unordered_map<std::string, std::vector<std::string>> adjacencyList;
	std::vector<std::string> order = {};

public:
	DatabaseMap(DatabaseConstructor &&databaseConstructor);
	int topoSort(const bool verbose = true);
	DatabaseIterator begin() { return DatabaseIterator(databaseConstructor, order); }
	DatabaseIterator end() { return DatabaseIterator(databaseConstructor, order); }
	TableConstructor at(std::string table);
};

struct QueryTemplateContainer
{
	/*
	* Initiates the queries to nullptr and might return nullptr queries, you need to check them manually
	*/
	std::map<std::string, std::string> queryTemplates;

	QueryTemplateContainer(DatabaseMap &databaseMap);
	
	void make_create_templates(DatabaseMap &databaseMap);

	void make_insert_templates(DatabaseMap &databaseMap);
	
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

int insert_experiments(sqlite3 *database, DatabaseMap &databaseMap, const bool verbose = true);
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

#endif