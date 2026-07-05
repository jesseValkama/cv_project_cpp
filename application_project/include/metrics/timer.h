#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <vector>

using time_point = std::chrono::high_resolution_clock::time_point;

class Timer
{
/*
* Acknowledgements:
*   https://www.youtube.com/watch?v=oEx5vGNFrLk
*/

private:
    time_point startTime;
    std::vector<time_point> checkpoints;
    bool saveCheckpoints;

public:
    Timer(const bool saveCheckpoints = false) :
        saveCheckpoints(saveCheckpoints) {};
    /*
    * Initiates the Timer class
    *
    * Args:
    *   saveCheckpoints: whether to save checkpoints in the checkpoints vector
    */

    void start();
    /*
    * Starts the timer
    */

    float checkpoint();
    /*
    * Creates a checkpoint
    *
    * Returns:
    *   float: the checkpoint (time in seconds compared to Timer::start())
    */

    std::vector<time_point> get_checkpoints();
    /*
    * Returns all stored checkpoints (vector)
    *
    * Retrns:
    *   std::vector<time_point>: vector of checkpoint
    */

    std::string format_time(float time);
    /*
    * formats time the HH:MM:SS format
    *
    * Args:
    *   time: time in seconds
    * 
    * Returns:
    *   std::string: time in hh:mm:ss
    */
};

#endif // TIMER_H