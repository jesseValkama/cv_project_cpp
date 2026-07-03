#include "database/query_template_manager.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sqlite3.h>
#include <string>
#include <string.h>
#include <vector>

#include "database/database_map.h"

int bind_values(sqlite3_stmt *&statement, const QueryTypes queryType, const ParamInfo &paramInfo, int &paramIdx)
{
	int ret = 0;
	if (paramInfo.primaryKey || queryType != QueryTypes::INSERT) { return 0; }
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

std::string escape_regex(const std::string &s)
{
    static const std::regex special{R"([.^$|()\\[*+?{\]])"};
    return std::regex_replace(s, special, R"(\$&)");
}

int ParamFormatter::make_template(std::string &t, const std::string &tableName, const TableConstructor &tableConstructor, const bool usePlaceholder, const bool includeTypes)
{
	int ret = 0;
	ret = this->prepare_template_list(tableName, tableConstructor, usePlaceholder, includeTypes);
	if (ret != 0) 
	{ 
		return ret; 
	}
	this->make_template_string(t, includeTypes);
	this->templateList.clear();
	int n = t.length();
	t = n < 2 ? "" : t.erase(n-2, n);
	t = "(" + t + ")";
	return 0;
}

int ParamFormatter::prepare_template_list(const std::string &tableName, const TableConstructor &tableConstructor, const bool usePlaceholder, const bool includeTypes)
{
	if (usePlaceholder && includeTypes)
	{
		return 1;
	}
	TemplateContainer templateContainer = TemplateContainer();
	int ret = 0;
	for (auto &[paramName, paramInfo] : tableConstructor)
	{
		if (!includeTypes && paramInfo.primaryKey)
		{
			continue;
		}
		templateContainer.attribute = usePlaceholder ? "?" : paramName;
		if (includeTypes)
		{
			ret = this->make_type_template(templateContainer, paramName, paramInfo);
			if (ret != 0) { return ret; }
		}
		this->add_params(templateContainer, paramInfo.primaryKey, paramInfo.foreignKey.has_value());
	}
	if (this->primaryKeys.size() > 1) 
	{ 
		std::cout << "Please use only a single primary key for " << tableName << std::endl;
		return 1; 
	}
	return 0;
}

int ParamFormatter::make_type_template(TemplateContainer &templateContainer, const std::string &paramName, const ParamInfo &paramInfo, const bool verbose)
{
	try
	{
		templateContainer.type = sqliteTypesMap.at(paramInfo.type);
	}
	catch (const std::out_of_range &e)
	{
		if (verbose) { std::cout << e.what() << std::endl;}
		return 1;
	}
	if (paramInfo.primaryKey)
	{
		templateContainer.type += " PRIMARY KEY";
	}
	if (paramInfo.foreignKey.has_value())
	{
		std::string type = "FOREIGN KEY (" + paramName + ") REFERENCES " + paramInfo.foreignKey->referenceTable + "(" + paramInfo.foreignKey->referenceColumn + ")";
		this->foreignKeys.emplace_back(TemplateContainer("", type, ""));
	}
	return 0;
}

void ParamFormatter::add_params(const TemplateContainer &templateContainer, const bool primaryKey, const bool foreignKey)
{
	if (primaryKey)
	{
		this->primaryKeys.emplace_back(std::move(templateContainer));
	}
	else
	{
		this->params.emplace_back(std::move(templateContainer));
	}
}

void ParamFormatter::make_template_string(std::string &t, const bool includeTypes)
{
	sort_vectors([](auto begin, auto end) { std::stable_sort(begin, end); }, this->params, this->foreignKeys);
	concatenate_vectors(this->templateList, this->primaryKeys, this->params, this->foreignKeys);
	for (TemplateContainer &p : this->templateList)
	{
		t += p.make_template(includeTypes) + ", "; 
	}
}

QueryTemplateManager::QueryTemplateManager(const db::DatabaseMap &databaseMap)
{
	(void) this->init_query_templates(this->queryTemplates, databaseMap);	
	this->paramFormatter = ParamFormatter();
}

void QueryTemplateManager::init_query_templates(std::unordered_map<std::string, std::string> &queryTemplates, const db::DatabaseMap &databaseMap)
{
	for (const auto &[tableName, tableInfo] : databaseMap)
	{
		queryTemplates[tableName];
	}
}

int QueryTemplateManager::make_templates(const QueryTypes queryType, const db::DatabaseMap &databaseMap)
{
	int ret = 0;
	for (const auto &[tableName, tableInfo] : databaseMap)
	{
		switch (queryType)
		{
			case QueryTypes::CREATE:
			{
				ret = this->make_create_templates(this->queryTemplates, tableName, tableInfo);
				break;
			}
			case QueryTypes::INSERT:
			{
				ret = this->make_insert_templates(this->queryTemplates, tableName, tableInfo);
				break;
			}
			case QueryTypes::VALIDATE:
			{
				ret = this->make_validate_templates(this->queryTemplates, tableName, tableInfo);
				break;
			}
			default:
			{
				std::cout << "Query type " << queryType << " is not supported yet" << std::endl;
				return 1;
			}
		}
	}
	return ret;
}

int QueryTemplateManager::validate_templates(const char *colName, const unsigned char *colText, const std::string &tableName, const TableConstructor &tableInfo) const
{
	if (strcmp(colName, "name") == 0)
	{
		return this->validate_template_name(colText, tableName);		
	}
	else if (strcmp(colName, "sql") == 0)
	{
		return this->validate_template_sql(colText, tableName, tableInfo);
	}
	return DB_ERR_UNKNOWN;
}

int QueryTemplateManager::make_create_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo)
{
	int ret = 0;
	std::string col;
	ret = this->paramFormatter.make_template(col, tableName, tableInfo, false, true);
	if (ret != 0) { return ret; }
	queryTemplates.at(tableName) = "CREATE TABLE " + tableName + " " + col + ";";
	col.clear();
	return 0;
}

int QueryTemplateManager::make_insert_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo)
{
	int ret = 0;
	std::string val;
	std::string col;
	ret = this->paramFormatter.make_template(val, tableName, tableInfo, true);
	if (ret != 0) { return ret; }
	ret = this->paramFormatter.make_template(col, tableName, tableInfo);
	if (ret != 0) { return ret; }
	queryTemplates.at(tableName) = "INSERT INTO " + tableName + " " + col + " VALUES " + val + ";";
	col.clear();
	val.clear();
	return 0;
}

int QueryTemplateManager::make_validate_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo)
{
	queryTemplates.at(tableName) = "SELECT name, sql FROM sqlite_master WHERE type='table' AND name = '" + tableName + "';";
	return 0;
}

int QueryTemplateManager::validate_template_name(const unsigned char *colText, const std::string &tableName) const
{
	if (strcmp(reinterpret_cast<const char *>(colText), tableName.c_str()) == 0)
	{
		return DB_OK;
	}
	std::cout << "Invalid table name: " << tableName << std::endl;
	return DB_ERR_SCHEMA;
}

int QueryTemplateManager::validate_template_sql(const unsigned char *colText, const std::string &tableName, const TableConstructor &tableInfo) const
{
	int ret = 0;
	std::string dbTemplates;
	ParamFormatter paramFormatter = ParamFormatter();
	ret = paramFormatter.make_template(dbTemplates, tableName, tableInfo, false, true);
	if (ret != DB_OK)
	{
		return ret;
	}
	std::string pattern = "^CREATE TABLE (\\w+) " + escape_regex(dbTemplates) + "$"; // do not ask me why there is no ; in colText
	const std::regex re = std::regex(pattern);
	std::smatch m;
	const std::string in = std::string(reinterpret_cast<const char *>(colText));
	if (std::regex_search(in, m, re))
	{
		return DB_OK;
	}
	std::cout << "Invalid sql for table " << tableName << std::endl;
	return DB_ERR_SCHEMA;
}