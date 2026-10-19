#include "train/grad_scaler.h"

#include <iostream>

#include <torch/torch.h>

amp::GradScaler::GradScaler(torch::Device dev, const double initScale, const double growthFactor, 
        const double backoffFactor, const int growthInterval, const bool enabled)
    : dev(dev), growthFactor(growthFactor), backoffFactor(backoffFactor),
            growthInterval(growthInterval), enabled(enabled)
{
    // TODO: error handling for bad inputs
    this->initScale = torch::tensor(initScale, c10::TensorOptions()
                        .dtype(torch::kFloat32).device(dev));
}

torch::Tensor amp::GradScaler::scale(const torch::Tensor &loss)
{
    if (!this->enabled)
    {
        return loss;
    }
    return loss * this->initScale.to(dev, true);
}

int amp::GradScaler::step(torch::optim::Optimizer &optimiser, const bool verbose)
{
        if (!this->enabled)
        {
            (void) optimiser.step();
            return AMP_OK;
        }
        this->unscale(optimiser);
        if (this->foundInf)
        {
            if (verbose) { std::cout << "Found an inf in the param settings" << std::endl; }
            return AMP_INF;
        }
        (void) optimiser.step();
        return AMP_OK;
}

void amp::GradScaler::update()
{
    // needs to be updated according to torch/amp/grad_scaler.py
    // currently chatgpt inspired 
    if (!this->enabled)
    {
        return;
    }
    if (this->foundInf)
    {
        this->initScale.mul_(backoffFactor);
        this->growthTracker = 0;
        return;
    }
    ++this->growthTracker;
    if (growthTracker >= growthInterval)
    {
        this->initScale.mul_(growthFactor);
        this->growthTracker = 0;
    }
}


void amp::GradScaler::unscale(torch::optim::Optimizer &optimiser)
{
    this->foundInf = false;
    torch::NoGradGuard no_grad;
    for (auto &group : optimiser.param_groups())
    {
        for (auto &param : group.params())
        {
            if (!param.grad().defined())
            {
                continue;
            }
            param.grad().div_(this->initScale);
            if (!torch::isfinite(param.grad()).all().item<bool>())
            {
                this->foundInf = true;
            }
        }
    }
}