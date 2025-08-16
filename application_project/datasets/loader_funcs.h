#ifndef LOADMNIST_H 
#define LOADMNIST_H

#include <stdint.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "../settings.h"

typedef std::vector<std::pair<int, char>> Info;
typedef torch::data::Example<> Example;

/*
* TODO:
* change functions from into return oriented
* improve safety
*/

/*
* Most of this file is inspired by the code from this post: 
* https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c 
* However, the code is modified (and safer?)
*/

uint32_t swap_endian(uint32_t val);
/*
* Helper function to decode information
* Credit: https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c 
*/

int check_magic(std::ifstream& fimg, std::ifstream& flabel, uint32_t labelMagic, uint32_t imgMagic, int len);
/*
* Helper function to check magic validity
* Potentially unsafe
* 
* Return indicates conditions:
*/

int check_labels(std::ifstream& fimg, std::ifstream& flabel, uint32_t& n, int len);
/*
* Helper function to check the number of labels and images
* Potentially unsafe
* 
* Return indicates conditions:
*/

int check_imgs(std::ifstream& fimg, std::ifstream& flabel, uint32_t& rows,
	uint32_t& cols, uint32_t minSize, uint32_t maxSize, int len);
/*
* Helper function to get img sizes
* Potentially unsafe
* 
* Return indicates conditions
*/

int load_mnist_info(MnistOpts& opts, Info& o, std::string type);
/*
* Naive function to read mnist info
* Expects the img size to ALWAYS be 28x28
* Potentially unsafe
* 
* Returns condition,
* 0: everything is fine
* 1: could not open a file
* 
* Crash:
* a fatal buffer overflow
*/

std::pair<cv::Mat, char> load_mnist_img(std::string path, size_t i, const Info& d, uint32_t rows, uint32_t cols);
/*
* Naive function to load mnist img
* Requires info from load_mnist_info
* 
* Throws runtime_error
* if either stream nor img could be opened
*/

#endif