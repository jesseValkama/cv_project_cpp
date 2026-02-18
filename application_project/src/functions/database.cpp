#include "functions/database.h"

#include <cassert>
#include <iostream>
#include <map>
#include <optional>
#include <sqlite3.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

std::string make_attribute_list_template(const TableConstructor &tableConstructor, const bool usePlaceholder, const bool includeTypes) // todo: include types
{
	assert(!(usePlaceholder && includeTypes));
	std::string attributeList = "";
	std::string attribute = "";
	size_t length = tableConstructor.size();
	size_t i = 1;
	for (auto &[paramName, paramInfo] : tableConstructor)
	{
		if (!paramInfo.primaryKey)
		{
			attribute = usePlaceholder ? "?" : paramName;
			attributeList += i < length ? attribute + ", " : attribute;
		}
		++i;
	}
	return "(" + attributeList + ")";
}

int bind_values(sqlite3_stmt *&statement, const ParamInfo &paramInfo, int &paramIdx)
{
	int ret = 0;
	if (paramInfo.primaryKey) { return 0; }
	switch (paramInfo.type)
	{
		case TYPE_INT: // todo: error handling
		{
			ret = sqlite3_bind_int(statement, paramIdx, std::get<int>(paramInfo.value));
			break;
		}
		case TYPE_REAL:
		{
			ret = sqlite3_bind_double(statement, paramIdx, std::get<double>(paramInfo.value));
			break;
		}
		case TYPE_TEXT:
		{
			ret = sqlite3_bind_text(statement, paramIdx, std::get<const char *>(paramInfo.value), strlen(std::get<const char *>(paramInfo.value)) + 1, SQLITE_STATIC);
			break;
		}
		case TYPE_BLOB:
		{
			ret = sqlite3_bind_blob(statement, paramIdx, std::get<const char *>(paramInfo.value), strlen(std::get<const char *>(paramInfo.value)), SQLITE_STATIC);
			break;
		}
		case TYPE_NULL:
		{
			ret = sqlite3_bind_null(statement, paramIdx);
			break;
		}
		default:
		{
			return 1;
		}
	}
	++paramIdx;
	return ret;
}

/*
* todo:
* seperate files
*/

DatabaseIterator &DatabaseIterator::operator++()
{
	++this->orderIdx;
	return *this;
}

DatabaseIterator &DatabaseIterator::operator--()
{
	--this->orderIdx;
	return *this;
}

DatabaseIterator DatabaseIterator::operator++(int)
{
	DatabaseIterator iter = *this;
	++(*this);
	return iter;
}

DatabaseIterator DatabaseIterator::operator--(int)
{
	DatabaseIterator iter = *this;
	++(*this);
	return iter;
}

std::pair<std::string, TableConstructor> DatabaseIterator::operator*()
{
	std::string tableName = this->tableOrder[this->orderIdx];
	return std::make_pair(tableName, this->databaseConstructor.at(tableName));
}

bool DatabaseIterator::operator==(const DatabaseIterator &other) const
{
	return this->orderIdx == tableOrder.size();
}

bool DatabaseIterator::operator!=(const DatabaseIterator &other) const
{
	return !(this->orderIdx == tableOrder.size());
}

DatabaseMap::DatabaseMap(DatabaseConstructor &&databaseConstructor)
	:	databaseConstructor(std::move(databaseConstructor))
{
	for (auto &[tableName, params] : this->databaseConstructor)
	{
		this->adjacencyList.insert({ tableName, {} });
	}
	for (auto &[tableName, params] : this->databaseConstructor)
	{
		for (auto &[paramName, paramInfo] : params)
		{
			if (!paramInfo.foreignKey.has_value())
			{
				continue;
			}
			this->adjacencyList.at(paramInfo.foreignKey->referenceTable).push_back(tableName);
		}
	}
}

int DatabaseMap::topoSort(const bool verbose)
{
	std::unordered_map<std::string, int> indeg;
	for (auto &[name, refs] : this->adjacencyList)
	{
		indeg.insert({name, 0});
	}
	for (auto &[name, refs] : this->adjacencyList)
	{
		for (auto &next : refs)
		{
			++indeg.at(next);
		}
	}
	std::queue<std::string> q;
	for (auto &[name, n] : indeg)
	{
		if (n == 0)
		{
			q.push(name);
		}
	}
	while (!q.empty())
	{
		std::string top = q.front();
		q.pop();
		this->order.push_back(top);
		for (std::string next : this->adjacencyList.at(top))
		{
			--indeg.at(next);
			if (indeg.at(next) == 0)
			{
				q.push(next);
			}
		}
	}
	if (this->order.size() != this->adjacencyList.size())
	{
		if (verbose) { std::cout << "loop detected in the database hierarchy" << std::endl; }
		return 1;
	}
	this->adjacencyList.clear();
	return 0;
}

TableConstructor DatabaseMap::at(std::string table)
{
	return this->databaseConstructor.at(table);
}

// start of spaghetti code, fix references, combine functions, and add const

QueryTemplateContainer::QueryTemplateContainer(DatabaseMap &databaseMap)
{
	for (auto [tableName, tableInfo] : databaseMap)
	{
		this->queryTemplates[tableName];
	}
}

void QueryTemplateContainer::make_create_templates(DatabaseMap &databaseMap)
{
	for (auto [tableName, tableInfo] : databaseMap)
	{
		std::string col = make_attribute_list_template(tableInfo);
		this->queryTemplates.at(tableName) = "CREATE TABLE ? " + col + ";";
	}
}

void QueryTemplateContainer::make_insert_templates(DatabaseMap &databaseMap)
{
	for (auto [tableName, tableInfo] : databaseMap) // todo fix this
	{
		std::string val = make_attribute_list_template(tableInfo, true);
		std::string col = make_attribute_list_template(tableInfo);
		this->queryTemplates.at(tableName) = "INSERT INTO " + tableName + " " + col + " VALUES " + val + ";";
	}
}

// end of spaghetti code

DatabaseConstructor db::get_database_constructor(const char *name, const char *model, const char *trainTime, const size_t epochs, const char *weights, 
							const double recall, const double precision, const double accuracy, const char *optimiser, const Settings &settings)
{
	return { {"experiments", {{"id", ParamInfo(nullptr, TYPE_INT, false, false, false, true)}, {"name", ParamInfo(name, TYPE_TEXT)}, {"date", ParamInfo("10.2.2026", TYPE_TEXT)}, {"model", ParamInfo(model, TYPE_TEXT)}, {"train_time", ParamInfo(trainTime, TYPE_TEXT)}}},
			{"metrics", {{"metrics_id", ParamInfo(nullptr, TYPE_INT, false, false, false, true, ForeignKey("experiments", "id"))}, {"weights", ParamInfo(weights, TYPE_TEXT)}, {"recall", ParamInfo(recall, TYPE_REAL)}, {"precision", ParamInfo(precision, TYPE_REAL)}, {"accuracy", ParamInfo(accuracy, TYPE_REAL)}}},
			{"config", {{"config_id", ParamInfo(nullptr, TYPE_INT, false, false, false, true, ForeignKey("experiments", "id"))}, {"optimiser", ParamInfo(optimiser, TYPE_TEXT)}, {"batch_size", ParamInfo(0, TYPE_INT)}, {"epochs", ParamInfo(0, TYPE_INT)}, {"valid_interval", ParamInfo(0, TYPE_INT)}, {"early_stop", ParamInfo(0, TYPE_INT)}}},
	}; // problems with size_t 
}

int db::open(const char *filename, sqlite3 **database, const bool verbose)
{
	int ret = sqlite3_open(filename, database);
	if (ret != SQLITE_OK && verbose)
	{
		std::cout << "could not open the database: " << sqlite3_errmsg(*database) << std::endl;
	}
	return ret;
}

int db::close(sqlite3 *database, const bool verbose)
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before calling close" << std::endl; }
		return 1;
	}
	int ret = sqlite3_close(database);
	if (ret != SQLITE_OK && verbose)
	{
		std::cout << "could not close the database: " << sqlite3_errmsg(database) << std::endl;
	}
	return ret;
}

int db::create_experiments(sqlite3 *database, const bool verbose)
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before creating experiment tables" << std::endl; }
		return 1;
	}
	// TODO: check if the tables already exist and make queries more modular
	const char *experiments = "CREATE TABLE experiments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, date TEXT NOT NULL, model TEXT NOT NULL, train_time TEXT NOT NULL);";
	const char *metrics = "CREATE TABLE metrics (metrics_id INTEGER PRIMARY KEY, weights TEXT, recall REAL NOT NULL, precision REAL NOT NULL, accuracy REAL NOT NULL, FOREIGN KEY (metrics_id) REFERENCES experiments(id));"; // weights ARE NOT unique as the user could overwrite some weights if they want to
	const char *config = "CREATE TABLE config (config_id INTEGER PRIMARY KEY, optimiser TEXT NOT NULL, batch_size INT NOT NULL, epochs INT NOT NULL, valid_interval INT NOT NULL, early_stop INT NOT NULL, FOREIGN KEY (config_id) REFERENCES experiments(id));";
	std::vector<const char *> query = { experiments, metrics, config };
	char *errmsg = nullptr;
	int ret = 0;
	for (const char *&q : query)
	{
		ret = sqlite3_exec(database, q, NULL, 0, &errmsg); // no user input -> unsafe is fine
		if (ret != 0)
		{
			if (verbose) { std::cout << "could not create experiments database" << errmsg << std::endl; }
			sqlite3_free(errmsg);
			return ret;
		}
	}
	return 0;
}

// fix const
int db::insert_experiments(sqlite3 *database, DatabaseMap &databaseMap, const bool verbose) // todo: implement force foreign keys and transactions
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before inserting into experiment tables" << std::endl; }
		return 1;
	}
	QueryTemplateContainer queryTemplateContainer(databaseMap);
	queryTemplateContainer.make_insert_templates(databaseMap);
	int ret = 0;
	int paramIdx = 1;
	for (auto [tableName, tableInfo] : databaseMap) // const and reference
	{
		paramIdx = 1;
		sqlite3_stmt *statement = nullptr;
		std::string query = queryTemplateContainer.queryTemplates[tableName];
		std::cout << query.c_str() << std::endl;
		ret = sqlite3_prepare_v2(database, query.c_str(), -1, &statement, NULL);
		if (ret != SQLITE_OK)
		{
			if (verbose) { std::cout << "could not prepare statement: " << sqlite3_errmsg(database) << std::endl; }
			return ret;
		}
		for (const auto &[paramName, paramInfo] : tableInfo)
		{
			ret = bind_values(statement, paramInfo, paramIdx); // fn auto increments the paramIdx
			if (ret != SQLITE_OK)
			{
				if (verbose) { std::cout << "could not bind values: " << sqlite3_errmsg(database) << std::endl; }
				sqlite3_finalize(statement);
				return ret;
			}
		}
		ret = sqlite3_step(statement);
		if (ret != SQLITE_DONE)
		{
			if (verbose) { std::cout << "could not execute the query: " << sqlite3_errmsg(database) << std::endl; }
			sqlite3_finalize(statement);
			return ret;
		}
		ret = sqlite3_finalize(statement);
		if (ret != SQLITE_OK)
		{
			if (verbose) { std::cout << "could not free the query: " << sqlite3_errmsg(database) << std::endl; }
			return ret;
		}
	}
	return 0;
}

/*
* the rest of the funcitons are not necessary
* and need to be either rewritten or deleted
*/ 

int unsafe_read(sqlite3 *database, const char *query, const bool verbose)
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before calling exec" << std::endl; }
		return 1;
	}
	sqlite3_stmt *statement = nullptr;
	int ret = sqlite3_prepare_v2(database, query, -1, &statement, NULL);
	if (ret != SQLITE_OK)
	{
		if (verbose) { std::cout << "could not prepare the statement: " << sqlite3_errmsg(database) << std::endl; }
		return ret;
	}
	int numCols = sqlite3_column_count(statement);
	bool done = false;
	int step = 0;
	while (!done)
	{
		step = sqlite3_step(statement);
		switch (step)
		{
			case SQLITE_ROW:
			{
				for (int i = 0; i < numCols; ++i)
				{
					const char *colName = sqlite3_column_name(statement, i);
					const unsigned char *text = sqlite3_column_text(statement, i);
					std::cout << "col: " << colName << "text: " << text << std::endl;
				}
				break;
			}
			case SQLITE_DONE:
			{
				ret = sqlite3_finalize(statement);
				done = true;
				break;
			}
			default: // not ideal, would preferably handle the other error cases, but at least won't result in a forever loop
			{
				std::cout << "the step failed: " << step << std::endl;
				done = true;
				ret = 1;
				break;
			}
		}
	}
	return ret;
}

const char *get_query(const Queries query)
{
	switch (query)
	{
	case CREATE:
		return "CREATE TABLE ? (?);";
	case INSERT:
		return "INSERT INTO test (?) VALUES (?);";
	case UPDATE:
		return "UPDATE test SET ? = ? WHERE id = ?;";
	}
	return nullptr;
}

int exec(sqlite3 *database, const Queries query, const char *param, const bool verbose)
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before calling exec" << std::endl; }
		return 1;
	}
	const char *q = get_query(query);
	if (!q) 
	{
		if (verbose) { std::cout << "enter a valid query" << std::endl; }
		return 1; 
	}
	sqlite3_stmt *statement = nullptr;
	int ret = sqlite3_prepare(database, q, -1, &statement, NULL);
	if (ret != SQLITE_OK)
	{
		if (verbose) { std::cout << "could not prepare the statement: " << sqlite3_errmsg(database) << std::endl; }
		return ret;
	}
	//const char *param = "; REMOVE FROM test WHERE id = 3;";
	ret = sqlite3_bind_blob(statement, 1, param, strlen(param) + 1, SQLITE_STATIC); // causes problems with types, use specific functions
	if (ret != SQLITE_OK)
	{
		if (verbose) { std::cout << "could not bind the parameter to the query: " << sqlite3_errmsg(database) << std::endl; }
		return ret;
	}
	ret = sqlite3_step(statement);
	if (ret != SQLITE_DONE) 
	{
		if (verbose) { std::cout << "could not execute the query: " << sqlite3_errmsg(database) << std::endl; }
		return ret;
	}
	ret = sqlite3_finalize(statement);
	return ret;
}

