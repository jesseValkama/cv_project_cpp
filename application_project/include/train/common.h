#ifndef TRAINCOMMON_H
#define TRAINCOMMON_H

#include <cstddef>
#include <limits>
#include <string>

namespace train
{

struct Tracker
{
	double trainLoss = std::numeric_limits<double>::infinity();
	double valLoss = std::numeric_limits<double>::infinity();
	double bestValLoss = std::numeric_limits<double>::infinity();
	double accuracy = 0.0;
	size_t epoch;
	std::string trainTime;
	std::string testTime;
};

} // end of train

#endif // TRAINCOMMON_H