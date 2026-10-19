#ifndef GRADSCALER_H
#define GRADSCALER_H

#include <torch/torch.h>

namespace amp
{

enum AmpRet
{
    AMP_OK = 0,
    AMP_INF = 10
};

class GradScaler
{

/*
* acknowledgements to
*   https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py
*   ChatGPT for helping me make sense of that python implementation
*/

private:

    torch::Device dev;
    torch::Tensor initScale;
    bool foundInf;
    double growthFactor;
    double backoffFactor;
    int growthInterval;
    int growthTracker;
    bool enabled;

public:

    GradScaler(torch::Device dev, const double initScale = 65536.0, const double growthFactor = 2.0, 
        const double backoffFactor = 0.5, const int growthInterval = 2000, const bool enabled = true);
    /*
    * todo: change the values to be GradScaler opts according to libtorch patterns
    */
        
    torch::Tensor scale(const torch::Tensor &loss);
    /*
    */

    int step(torch::optim::Optimizer &optimiser, const bool verbose = true);
    /*
    */
    

    void update();
    /*
    */

private:

    void unscale(torch::optim::Optimizer &optimiser);
    /*
    */

};

}

#endif // GRADSCALER_H