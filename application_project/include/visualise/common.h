#ifndef VISUALISECOMMON_H
#define VISUALISECOMMON_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <optional>

#include "settings/settings.h"

namespace fmvis
{

struct Visualisers
{
    VisualiserTasks task;
    cv::ColormapTypes cmap;
    std::optional<int16_t> idx;
    Visualisers(VisualiserTasks task, cv::ColormapTypes cmap, std::optional<int> idx = std::nullopt) 
                : task(task), cmap(cmap), idx(idx) {};
};

struct FeatureMapVisualiserOptions
{
    float imgWeight;
    std::optional<std::string> savepath;
    std::optional<Visualisers> task;
    FeatureMapVisualiserOptions(float imgWeight, std::optional<std::string> savepath)
        : imgWeight(imgWeight), savepath(savepath) {};
};

} // end of fmvis

#endif // VISUALISECOMMON_H