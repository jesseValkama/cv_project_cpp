#include "loader_funcs.h"

#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

uint32_t swap_endian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

int check_magic(std::ifstream &fimg, std::ifstream &flabel, uint32_t labelMagic, uint32_t imgMagic, int len)
{
	uint32_t magic = 0;
	fimg.read(reinterpret_cast<char*>(&magic), len);
	magic = swap_endian(magic);
	if (magic != imgMagic)
	{
		std::cout << "The img magic is incorrect" << "\n";
		return 1;
	}
	flabel.read(reinterpret_cast<char*>(&magic), len);
	magic = swap_endian(magic);
	if (magic != labelMagic)
	{
		std::cout << "The label magic is incorrect" << "\n";
		return 2;
	}
	return 0;
}

int check_labels(std::ifstream &fimg, std::ifstream &flabel, uint32_t &n, int len)
{
	uint32_t nImgs = 0, nLabels = 0;
	fimg.read(reinterpret_cast<char*>(&nImgs), len);
	flabel.read(reinterpret_cast<char*>(&nLabels), len);
	nImgs = swap_endian(nImgs);
	nLabels = swap_endian(nLabels);
	if (nImgs != nLabels)
	{
		std::cout << "n of labels doesn't correspond the the n of imgs" << "\n";
		return 1;
	}
	n = nImgs;
	return 0;
}

int check_imgs(std::ifstream& fimg, std::ifstream& flabel, uint32_t &rows, 
	uint32_t &cols, uint32_t minSize, uint32_t maxSize, int len)
{
	fimg.read(reinterpret_cast<char*>(&rows), len);
	fimg.read(reinterpret_cast<char*>(&cols), len);
	rows = swap_endian(rows);
	cols = swap_endian(cols);
	uint32_t size = rows * cols;
	if (size < minSize || size > maxSize)
	{
		std::cout << "Weird img size" << "\n";
		return 1;
	}
	
	return 0;
}

int load_mnist_info(MnistOpts &opts, Info &o, std::string type)
{
	assert(type == "train" || type == "test");
	std::string fImgs = (type == "train") ? opts.fTrainImgs : opts.fTestImgs;
	std::string fLabels = (type == "train") ? opts.fTrainLabels : opts.fTestLabels;

	std::ifstream fimg(fImgs, std::ios::in | std::ios::binary);
	std::ifstream flabel(fLabels, std::ios::in | std::ios::binary);

	if (!(flabel.is_open() && fimg.is_open()))
	{
		std::cout << "Could not open the stream" << "\n";
		return 1;
	}

	uint32_t n = 0, rows = 0, cols = 0;
	int status = 0;
	status = check_magic(fimg, flabel, 2049, 2051, 4);
	if (status != 0)
	{
		return status;
	}
	status = check_labels(fimg, flabel, n, 4);
	if (status != 0)
	{
		return status;
	}
	status = check_imgs(fimg, flabel, rows, cols, 784, 784, 4);
	if (status != 0)
	{
		return status;
	}
	
	char label;
	int pos = 0;
	int size = static_cast<int>(rows) * static_cast<int>(cols);
	// naive since, it expects all images to be the same size
	for (int i = 0; i < n; ++i)
	{
		flabel.read(&label, 1);
		pos = fimg.tellg();
		o.push_back(std::make_pair(pos, label));
		pos += size;
		fimg.seekg(pos, std::ios::beg);
	}

	return 0;
}

std::pair<cv::Mat, char> load_mnist_img(std::string path, size_t i, const Info &d, uint32_t rows, uint32_t cols)
{
	std::ifstream fimg(path, std::ios::in | std::ios::binary);
	if (!fimg.is_open())
	{
		std::runtime_error("Could not open the stream");
	}

	uint32_t imgsz = rows * cols;
	int pos = d[i].first;
	char l = d[i].second;
	std::vector<char> buf(imgsz);
	fimg.seekg(pos, std::ios::beg);
	fimg.read(buf.data(), buf.size());
	cv::Mat img(rows, cols, CV_8UC1, buf.data());
	if (img.empty())
	{
		std::runtime_error("Could not open the img");
	}

	return std::make_pair(img.clone(), l);
}
