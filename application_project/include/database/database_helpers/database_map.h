#ifndef DATABASEMAP_H
#define DATABASEMAP_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/database_helpers/param_info.h"

using TableConstructor = std::unordered_map<std::string, ParamInfo>;
using DatabaseConstructor = std::unordered_map<std::string, TableConstructor>;

class DatabaseIteratorBase
{
	DatabaseIteratorBase() {};
};

class DatabaseIterator
{
	/*
	* this class is heavilty inspired the tutorial:
	* https://www.youtube.com/watch?v=F9eDv-YIOQ0
	*/
	 DatabaseConstructor databaseConstructor;
	 std::vector<std::string> tableOrder;
	 int orderIdx = 0;
public:
	DatabaseIterator(DatabaseConstructor &databaseConstructor, std::vector<std::string> &tableOrder) 
		: databaseConstructor(databaseConstructor), tableOrder(tableOrder) {};
	DatabaseIterator &operator++();
	DatabaseIterator &operator--();
	DatabaseIterator operator++(int);
	DatabaseIterator operator--(int);
	std::pair<std::string, TableConstructor> operator*();
	bool operator==(const DatabaseIterator &other) const;
	bool operator!=(const DatabaseIterator &other) const;
};

class ConstDatabaseIterator
{
	const DatabaseConstructor databaseConstructor;
	const std::vector<std::string> tableOrder;
	int orderIdx = 0;
public:
	ConstDatabaseIterator(const DatabaseConstructor &databaseConstructor, const std::vector<std::string> &tableOrder)
		: databaseConstructor(databaseConstructor), tableOrder(tableOrder) {};
	ConstDatabaseIterator &operator++();
	ConstDatabaseIterator &operator--();
	ConstDatabaseIterator operator++(int);
	ConstDatabaseIterator operator--(int);
	std::pair<std::string, TableConstructor> operator*();
	bool operator==(const ConstDatabaseIterator &other) const;
	bool operator!=(const ConstDatabaseIterator &other) const;
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
	ConstDatabaseIterator begin() const { return ConstDatabaseIterator(databaseConstructor, order); }
	ConstDatabaseIterator end() const { return ConstDatabaseIterator(databaseConstructor, order); }
	TableConstructor at(std::string table);
};

#endif // DATABASEMAP_H
