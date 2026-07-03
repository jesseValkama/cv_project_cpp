#ifndef DATABASEMAP_H
#define DATABASEMAP_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/param_info.h"

using TableConstructor = std::unordered_map<std::string, ParamInfo>;
using DatabaseConstructor = std::unordered_map<std::string, TableConstructor>;

/*
* this tutorial was helpful for the iterators
* https://www.youtube.com/watch?v=F9eDv-YIOQ0
*/

template<typename T>
class DatabaseIteratorBase
/*
* The idea to use crtp comes from ChatGPT when i asked if there was
* any cpp syntax that would let me use inheritance here instead typing
* the code twice
*/
{
protected:
	int orderIdx = 0;
public:
	DatabaseIteratorBase(int orderIdx = 0) 
		: orderIdx(orderIdx) {}
	T &operator++()
	{
		++this->orderIdx;
		return static_cast<T&>(*this);
	}
	T &operator--()
	{
		--this->orderIdx;
		return static_cast<T&>(*this);
	}
	T operator++(int)
	{
		T iter = static_cast<T>(*this);
		++(*this);
		return iter;
	}
	T operator--(int)
	{
		T iter = static_cast<T>(*this);
		--(*this);
		return iter;
	}
	int operator*() 
	{
		return orderIdx;
	}
};

class DatabaseIterator : public DatabaseIteratorBase<DatabaseIterator>
{
	 std::vector<std::pair<std::string, TableConstructor>> tableOrder;
public:
	DatabaseIterator(int orderIdx, std::vector<std::pair<std::string, TableConstructor>> &tableOrder) 
		: DatabaseIteratorBase<DatabaseIterator>(orderIdx), tableOrder(tableOrder) {};
	std::pair<std::string, TableConstructor> operator*()
	{
		return this->tableOrder[this->orderIdx];
	}
	bool operator==(const DatabaseIterator &other) const
	{
		return this->orderIdx == other.orderIdx;
	}
	bool operator!=(const DatabaseIterator &other) const
	{
		return !(this->orderIdx == other.orderIdx);
	}
};

class ConstDatabaseIterator : public DatabaseIteratorBase<DatabaseIterator>
{
	const std::vector<std::pair<std::string, TableConstructor>> tableOrder;
public:
	ConstDatabaseIterator(int orderIdx, const std::vector<std::pair<std::string, TableConstructor>> &tableOrder)
		: DatabaseIteratorBase<DatabaseIterator>(orderIdx), tableOrder(tableOrder) {};
	std::pair<std::string, TableConstructor> operator*()
	{
		return this->tableOrder[this->orderIdx];
	}
	bool operator==(const ConstDatabaseIterator &other) const
	{
		return this->orderIdx == other.orderIdx;
	}
	bool operator!=(const ConstDatabaseIterator &other) const
	{
		return !(this->orderIdx == other.orderIdx);
	}
};

namespace db
{

class DatabaseMap
{
	/*
	* acknowledgements:
	*	https://www.geeksforgeeks.org/python/introduction-to-graphs-in-python/
	*	https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
	*/
private:
	DatabaseConstructor databaseConstructor;
	std::unordered_map<std::string, std::vector<std::string>> adjacencyList;
	std::vector<std::pair<std::string, TableConstructor>> tableOrder;

public:
	DatabaseMap(DatabaseConstructor &&databaseConstructor);
	
	int topo_sort(const bool verbose = true);
	
	// uses a hacky way of checking if at the end
	DatabaseIterator begin() { return DatabaseIterator(0, tableOrder); }
	DatabaseIterator end() { return DatabaseIterator(tableOrder.size(), tableOrder); }
	ConstDatabaseIterator begin() const { return ConstDatabaseIterator(0, tableOrder); }
	ConstDatabaseIterator end() const { return ConstDatabaseIterator(tableOrder.size(), tableOrder); }
};

} // end of db

#endif // DATABASEMAP_H
