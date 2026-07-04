#include "metrics/timer.h"

#include <iomanip>
#include <sstream>
#include <string>

void Timer::start()
{
    this->startTime = std::chrono::high_resolution_clock::now();
}

float Timer::checkpoint()
{
    std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - this->startTime;
    return duration.count();
}

std::vector<time_point> Timer::get_checkpoints()
{
    return this->checkpoints;
}


std::string Timer::format_time(float time)
{
    // sadly no c++ 20
    int hours   = static_cast<int>(time) / 3600;
    int minutes = (static_cast<int>(time) % 3600) / 60;
    int seconds    = static_cast<int>(time) % 60;
    // credit chatgpt
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << seconds;
    return oss.str();
}