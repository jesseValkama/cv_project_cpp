#ifndef LOADMNIST_H 
#define LOADMNIST_H

#include <fstream>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "../settings.h"

typedef std::vector<std::pair<std::streampos, char>> Info;
typedef torch::data::Example<> Batch;

/*
* MNIST loader funcs are either copied or inspired by this:
* https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c 
* However, the code is modified (and safer?)
*/

uint32_t swap_endian(uint32_t val);
/*
* Swaps the endian either from big->little or little->big (for 32bit)
* The mnist dataset is big-endian, this fn changes it to be little endian
* Credit: https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c
* 
* Args:
*	val: the value to decode
* 
* Returns:
*	the decoded value
*/

int check_magic(std::ifstream& fimg, std::ifstream& flabel, uint32_t labelMagic, uint32_t imgMagic, int len);
/*
* Helper function to check magic validity
* Potentially unsafe
* 
* Args:
*	fimg: stream for imgs
*	flabel: stream for labels
*	labelMagic: label magic num
*	imgMagic: img magic num
*	len: n bytes to read
*
* Returns: 
*	0: successful
*	1: failed (logged to terminal)
*/

int check_labels(std::ifstream &fimg, std::ifstream &flabel, uint32_t &n, int len);
/*
* Helper function to check the number of labels and images
* Potentially unsafe
* 
* Args:
*	fimg: stream for imgs
*	flabel: stream for labels
*	n: container for the amount of labels and imgs in the file
*	len: n of bytes to read from the stream
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int check_imgs(std::ifstream& fimg, std::ifstream& flabel, uint32_t& rows,
	uint32_t& cols, uint32_t minSize, uint32_t maxSize, int len);
/*
* Helper function to get img sizes and to validate the size
* Potentially unsafe
* 
* Args:
*	fimg: stream for imgs
*	flabel: stream for labels
*	rows: n rows for each img
*	cols: n rows for each label
*	minSize: expected min size for the imgs
*	maxSize: expected max size for the imgs
*	len: the n of bytes to read from the stream
* 
* Returns: 
*	0: successful
*	1: failed (logged to terminal)
*/

int load_mnist_info(DatasetOpts& opts, Info& o, std::string type);
/*
* Naive function to read mnist info
* Expects the img size to ALWAYS be 28x28
* stores info in vector of relative position and label
* Potentially unsafe
* 
* Args:
*	opts: options for training
*	info: the container where to store the relative positions
*	of each label and img
*	type: test or train, used to determine which file to open
* 
* Returns:
*	0: successful
*	1: failed (logged to terminal)
*/

int load_cifar10_batch_info(std::ifstream &stream, Info &o, uint32_t n, uint32_t imgsz = 3072, uint32_t labelsz = 1);
/*
* Naive function to read cifar10 data
* Instructions for loading data:
*	https://www.cs.toronto.edu/%7Ekriz/cifar.html
* 
* Args:
*	stream: the open stream for a batch
*	o: the vector to load the indfo to
*	bs: the amount of label, img pairs in the batch
*	imgsz: image size (w*h*d) in bytes
*	labelsz: label size in bytes
* 
* Returns:
*	0: succesful
*	1: failed (logged to terminal)
*/

int load_cifar10_info(DatasetOpts &opts, Info &o, std::string type, uint32_t bs = 10000);
/*
* Function to combine cifar-10 data (by default train is divided into 5 batches)
* 
* Args:
*	opts: options for cifar-10
*	o: the vector to read all the defined batches to (defined in settings.h)
*	type: bool whether "train" or "test"
*	bs: the amount of label, img pairs in a batch
* 
* Returns:
*	0: succesful
*	1: failed (logged to terminal)
*/

std::optional<cv::Mat> load_png(std::string path, int nc, int imgresz = -1);
/*
* Loads an img from path (and potentially resizes it)
* 
* Args:
*	path: path to the img
*	nc: number of channels for the img
*	imgresz: the size to resize the img to (-1 for no resize)
* 
* Returns:
*	img: successful 
*	nullopt: failed (logged to terminal)
*/

std::optional<std::pair<cv::Mat, char>> load_mnist_img(std::string path, size_t i, const Info& d, uint32_t rows, uint32_t cols, int imgresz = -1, uint32_t depth = 1, bool channelWise = false);
/*
* Naive function to load mnist img and label
* Requires info from load_mnist_info
* RGB imgs are returned as bgr
* 
* Args:
*	path: path to the stream
*	i: index of the vector for the img + label
*	d: container for the imgs and labels (vector of pairs)
*	rows: n of rows for the img
*	cols: n of cols for the img
*	imgresz: the size to resize img (-1 for no resize)
* 
* Returns:
*	pair<img, char>: successful (img and label)
*	nullopt: failed
*/

torch::Tensor greyscale2Tensor(cv::Mat img, int imgsz, int div = -1);
/*
* Loads a greyscale cv::Mat into a Tensor (normalises it by dividing with div, -1 to skip, but doesn't include z scaling)
* 
* Args:
*	img: the input img
*	imgsz: the size of the img
*	div: max value for img (usually 255) for normalisation (-1 to skip normalisation)
* 
* Returns:
*	tensor: the output img
*/

torch::Tensor mat2Tensor(cv::Mat &img, int imgsz, int nc, int div = -1, bool toRGB = false);
/*
* Loads an rgb cv::Mat into a Tensor (normalises it by dividing with div, -1 to skip, but doesn't include z scaling)
*
* Args:
*	img: the input img
*	imgsz: the size of the img
*	div: max value for img (usually 255) for normalisation (-1 to skip normalisation)
* 
* Returns:
*	Tensor: the output img
*/

std::optional<cv::Mat> Tensor2mat(torch::Tensor timg, int squeeze = -1, bool toBGR = false, std::pair<std::vector<double>, std::vector<double>> scale = { {}, {} }, float imgMax = 255.0f);
/*
* Loads a tensor to a cv::Mat.
* Denormalisation is also possible, default scale and imgMax values skip dernomalisation
* 
* Args:
*	timg: img as a tensor
*	squeeze: the index to squeeze from -1 to skip 
*	toBGR: whether to transform from rgb2bgr or not
*	scale: 1st mean and 2nd standard deviation for z-scaling
*	imgMax: the max value for a pixel (needs to be <= 255), used in denormalising the image
* 
* Returns:
*	img: output cv::Mat
*	nullopt: failed (logged to terminal)
*/

void z_scale_Tensor(torch::Tensor &timg, std::pair<std::vector<double>, std::vector<double>> scale);
/*
* Performs z-scaling on a tensor
* 
* Args:
*	timg: the input tensor
*	scale: pair of vector of means and stdevs
*/

#endif