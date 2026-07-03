#ifndef QUERYTEMPLATEMANAGER_H
#define QUERYTEMPLATEMANAGER_H

#include <algorithm>
#include <functional>
#include <sqlite3.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/database_map.h"

int bind_values(sqlite3_stmt *&statement, const QueryTypes queryType, const ParamInfo &paramInfo, int &idx);
/*
*/


std::string escape_regex(const std::string &s);
/*
* credit chatgpt for this function
*/

template<typename T, typename... Args>
void concatenate_vectors(std::vector<T> &out, Args &...args)
/*
* This funcions in mostly created by ChatGPT when i asked if i could use python *args in cpp
*/
{
	out.reserve((args.size() + ...));
	(out.insert(out.end(), 
		std::make_move_iterator(args.begin()), 
		std::make_move_iterator(args.end())), ...);
	(args.clear(), ...);
}

template<typename Func, typename... Args>
void sort_vectors(Func func, Args &...args)
{
	(func(args.begin(), args.end()), ...);
}


struct TemplateContainer
{
	std::string attribute;
	std::string type;
	std::string delim;

	TemplateContainer(const std::string &attribute = "", const std::string &type = "", const std::string &delim = " ")
		: attribute(attribute), type(type), delim(delim) {}

	std::string make_template(const bool includeTypes)
	{
		std::string t = this->attribute;
		if (includeTypes)
		{
			t += this->delim + this->type;
		}
		return t;
	}

	bool operator<(const TemplateContainer &other) const
	{
		return this->attribute < other.attribute;
	}
};

class ParamFormatter
{

private:

	std::vector<TemplateContainer> primaryKeys;
	std::vector<TemplateContainer> foreignKeys;
	std::vector<TemplateContainer> params;
	std::vector<TemplateContainer> templateList;

public:

	ParamFormatter() {}

	int make_template(std::string &t, const std::string &tableName, const TableConstructor &tableConstructor, const bool usePlaceholder = false, const bool includeTypes = false);

private:

	int prepare_template_list(const std::string &tableName, const TableConstructor &tableConstructor, const bool usePlaceholder = false, const bool includeTypes = false);

	int make_type_template(TemplateContainer &templateContainer, const std::string &paramName, const ParamInfo &paramInfo, const bool verbose = true);

	void add_params(const TemplateContainer &templateContainer, const bool primaryKey, const bool foreignKey);

	void make_template_string(std::string &t, const bool includeTypes);

};

class QueryTemplateManager
{
	/*
	* Initiates the queries to nullptr and might return nullptr queries, you need to check them manually
	*/
private:
	ParamFormatter paramFormatter;

public:
	std::unordered_map<std::string, std::string> queryTemplates;
	/*
	*/

	QueryTemplateManager(const db::DatabaseMap &databaseMap);
	/*
	*/

	int make_templates(const QueryTypes queryType, const db::DatabaseMap &databaseMap);
	/*
	*/

	int validate_templates(const char *colName, const unsigned char *colText, const std::string &tableName, const TableConstructor &tableInfo) const;
	/*
	*/
	
private:
	void init_query_templates(std::unordered_map<std::string, std::string> &queryTemplates, const db::DatabaseMap &databaseMap);
	/*
	*/

	int make_create_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo);
	/*
	*/

	int make_insert_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo);
	/*
	*/

	int make_validate_templates(std::unordered_map<std::string, std::string> &queryTemplates, const std::string &tableName, const TableConstructor &tableInfo);
	/*
	* acknowledgement:
	* 	https://stackoverflow.com/questions/1601151/how-do-i-check-in-sqlite-whether-a-table-exists
	*/

	int validate_template_name(const unsigned char *colText, const std::string &tableName) const;
	/*
	*/

	int validate_template_sql(const unsigned char *colText, const std::string &tableName, const TableConstructor &tableInfo) const;
	/*
	*/
	
};

#endif // QUERYTEMPLATEMANAGER_H