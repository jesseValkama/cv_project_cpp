#ifndef EXPERIMENTSDATABASE_H
#define EXPERIMENTSDATABASE_H

#include <cstddef>

int insert_experiments(const char *filename, const char *experimentName, const char *modelName, const char *datasetName, const char *trainTime, 
	const size_t epochs, const char *weightsName, const double recall, const double precision, const double accuracy, 
	const char *optimiserName);

#endif // EXPERIMENTSDATABASE_H