#ifndef GETMODEL_H
#define GETMODEL_H

#include <memory>
#include <optional>
#include <string>

#include <torch/torch.h>

#include "models/lenet.h"
#include "models/resnet.h"
#include "settings/settings.h"

namespace nn = torch::nn;

template<typename Model>
Model get_model(const ModelTypes modelType, const DatasetOpts &dsOpts, const std::optional<std::string> &weightsPath, const bool fmvis = true)
{
    switch (modelType)
    {
        case LENET:
        {
            Model model = LeNet(dsOpts.imgresz, dsOpts.numOfClasses, dsOpts.numOfChannels, fmvis), weightsPath;
            if (weightsPath.has_value())
            {
                torch::load(model, *weightsPath);
            }
            return model;
        }
        case RESNET:
        {
            Model model  = ResNet(dsOpts.imgresz, dsOpts.numOfClasses, dsOpts.numOfChannels, fmvis), weightsPath;
            if (weightsPath.has_value())
            {
                torch::load(model, *weightsPath);
            }
            return model;
        }
        default:
        {
            return nullptr;
        }
    }
}

#endif // GETMODEL_H