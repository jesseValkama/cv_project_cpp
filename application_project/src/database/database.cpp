#include "database/database.h"

#include <cassert>
#include <cstring>
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

#include "database/param_info.h"
#include "database/database_map.h"
#include "database/query_template_manager.h"
#include "database/database_helpers.h"

std::unordered_map<SqliteTypes, std::string> sqliteTypesMap = 
{
	{SqliteTypes::TYPE_INT, "INTEGER"}, 
	{SqliteTypes::TYPE_REAL, "REAL"},
	{SqliteTypes::TYPE_TEXT, "TEXT"},
	{SqliteTypes::TYPE_BLOB, "BLOB"},
	{SqliteTypes::TYPE_NULL, "NULL"}
};

DatabaseConstructor db::get_database_constructor(const char *name, const char *model, const char *dataset, const char *trainTime, const size_t epochs, const char *weights, 
							const double recall, const double precision, const double accuracy, const char *optimiser)
{
	return { {"experiments", {{"id", ParamInfo(nullptr, TYPE_INT, false, false, false, true)}, {"name", ParamInfo(name, TYPE_TEXT)}, {"date", ParamInfo("10.2.2026", TYPE_TEXT)}, {"model", ParamInfo(model, TYPE_TEXT)}, {"dataset", ParamInfo(dataset, TYPE_TEXT)}, {"train_time", ParamInfo(trainTime, TYPE_TEXT)}}},
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
	database = nullptr; // memory leak?
	return ret;
}

int db::exec(sqlite3 *database, QueryTypes queryType, const DatabaseMap &databaseMap, const bool verbose)
{
	if (!database)
	{
		if (verbose) { std::cout << "make sure to open the database before calling exec" << std::endl; }
		return 1;
	}
	int ret = 0;
	QueryTemplateManager queryTemplateManager(databaseMap);
	ret = queryTemplateManager.make_templates(queryType, databaseMap);
	if (ret != 0) {return ret;}
	int paramIdx = 1;
	sqlite3_stmt *stmt = nullptr;
	std::string query;
	for (const auto &[tableName, tableInfo] : databaseMap)
	{
		paramIdx = 1;
		query = queryTemplateManager.queryTemplates[tableName];
		ret = prepare_stmt(database, stmt, query.c_str(), verbose);
		if (ret != DB_OK)
		{
			return ret;
		}
		for (const auto &[paramName, paramInfo] : tableInfo)
		{
			ret = bind_stmt(database, stmt, queryType, paramInfo, paramIdx);
			if (!(ret == DB_OK || ret == DB_SKIP))
			{
				return ret;
			}
		}
		ret = step_stmt(database, stmt, 0, queryTemplateManager, tableName, tableInfo, verbose);
		if (ret != DB_OK)
		{
			return ret;
		}
		ret = free_stmt(database, stmt, verbose);
		if (ret != DB_OK)
		{
			return ret;
		}
	}
	return DB_OK;
}