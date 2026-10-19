#include "datasets/inference.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "augmentations/geometry.h"
#include "settings/settings.h"
#include "datasets/loader_funcs.h"

namespace fs = std::filesystem;

Batch InferenceDataset::get(size_t i)
{
	std::optional<cv::Mat> img = load_png(this->imgPaths[i], this->dsOpts.numOfChannels, this->dsOpts.imgresz);
    if (!img.has_value())
    {
        return {torch::tensor({}), torch::tensor({})};
    }
    fs::path filePath(this->imgPaths[i]);
    std::string fname = filePath.filename().string();
    bool toRGB = (this->dsOpts.numOfChannels == 3) ? true : false;
    torch::Tensor timg = mat2Tensor(*img, this->dsOpts.imgresz, this->dsOpts.numOfChannels, 255, toRGB);
    if (dsOpts.async)
    {
        timg = timg.pin_memory();
    }
    return {timg, torch::tensor({})};
}

torch::optional<size_t> InferenceDataset::size() const
{
	return this->imgPaths.size();
}
