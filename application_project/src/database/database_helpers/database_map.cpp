#include "database/database_helpers/database_map.h"

#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/database_helpers/param_info.h"

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
	return std::make_pair(tableName, this->databaseConstructor.at(tableName)); // not compatible with &
}

bool DatabaseIterator::operator==(const DatabaseIterator &other) const
{
	return this->orderIdx == tableOrder.size();
}

bool DatabaseIterator::operator!=(const DatabaseIterator &other) const
{
	return !(this->orderIdx == tableOrder.size());
}

// start of repeated code, needs to become more modular

ConstDatabaseIterator &ConstDatabaseIterator::operator++()
{
	++this->orderIdx;
	return *this;
}

ConstDatabaseIterator &ConstDatabaseIterator::operator--()
{
	--this->orderIdx;
	return *this;
}

ConstDatabaseIterator ConstDatabaseIterator::operator++(int)
{
	ConstDatabaseIterator iter = *this;
	++(*this);
	return iter;
}

ConstDatabaseIterator ConstDatabaseIterator::operator--(int)
{
	ConstDatabaseIterator iter = *this;
	++(*this);
	return iter;
}

std::pair<std::string, TableConstructor> ConstDatabaseIterator::operator*()
{
	std::string tableName = this->tableOrder[this->orderIdx];
	return std::make_pair(tableName, this->databaseConstructor.at(tableName));
}

bool ConstDatabaseIterator::operator==(const ConstDatabaseIterator &other) const
{
	return this->orderIdx == tableOrder.size();
}

bool ConstDatabaseIterator::operator!=(const ConstDatabaseIterator &other) const
{
	return !(this->orderIdx == tableOrder.size());
}

// end of repeated code

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
