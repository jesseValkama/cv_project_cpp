#include "functions/experiments_database.h"

#include <iostream>
#include <sqlite3.h>

#include "database/database.h"
#include "database/database_map.h"

int insert_experiments(const char *filename, const char *experimentName, const char *modelName, const char *datasetName, const char *trainTime, 
	const size_t epochs, const char *weightsName, const double recall, const double precision, const double accuracy, 
	const char *optimiserName)
{
	sqlite3 *database = nullptr;
	int ret = db::open(filename, &database);
	if (ret != DB_OK) { return ret; }
	db::DatabaseMap databaseMap(db::get_database_constructor(experimentName, modelName, datasetName, trainTime, epochs, weightsName, 
																recall, precision, accuracy, optimiserName));
	ret = databaseMap.topo_sort();
	if (ret != DB_OK) 
	{ 
		db::close(database);
		return ret; 
	}
	(void) db::exec(database, QueryTypes::CREATE, databaseMap);
	ret = db::exec(database, QueryTypes::VALIDATE, databaseMap);
	if (ret != DB_OK) 
	{ 
		db::close(database);
		return ret;
	}
	ret = db::exec(database, QueryTypes::INSERT, databaseMap);
	if (ret != DB_OK) 
	{ 
		db::close(database);
		return ret;
	}
	ret = db::close(database);
	if (ret != DB_OK) 
	{ 
		return ret; 
	}
	return DB_OK;
}