#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <vector>

using time_point = std::chrono::high_resolution_clock::time_point;

class Timer
{
private:
    time_point startTime;
    std::vector<time_point> checkpoints;
    bool saveCheckpoints;
public:
    Timer(const bool saveCheckpoints = false) :
        saveCheckpoints(saveCheckpoints) {};
    void start();
    float checkpoint();
    std::vector<time_point> get_checkpoints();
    std::string format_time(float time);
};

#endif // TIMER_H