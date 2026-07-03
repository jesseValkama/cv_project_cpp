#include "database/database_map.h"

#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/param_info.h"

db::DatabaseMap::DatabaseMap(DatabaseConstructor &&databaseConstructor)
		: databaseConstructor(std::move(databaseConstructor))
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

int db::DatabaseMap::topo_sort(const bool verbose)
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
	std::string top;
	while (!q.empty())
	{
		top = q.front();
		q.pop();
		this->tableOrder.push_back(std::make_pair(top, this->databaseConstructor.at(top)));
		for (std::string next : this->adjacencyList.at(top))
		{
			--indeg.at(next);
			if (indeg.at(next) == 0)
			{
				q.push(next);
			}
		}
	}
	if (this->tableOrder.size() != this->adjacencyList.size())
	{
		if (verbose) { std::cout << "loop detected in the database hierarchy" << std::endl; }
		return 1;
	}
	this->databaseConstructor.clear();
	this->adjacencyList.clear();
	return 0;
}