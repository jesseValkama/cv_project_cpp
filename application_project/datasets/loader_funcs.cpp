#include "loader_funcs.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdint.h>
#include <string>
#include <tuple>
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
	if (!fimg)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

	magic = swap_endian(magic);
	if (magic != imgMagic)
	{
		std::cout << "The img magic is incorrect" << "\n";
		return 1;
	}

	flabel.read(reinterpret_cast<char*>(&magic), len);
	if (!flabel)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

	magic = swap_endian(magic);
	if (magic != labelMagic)
	{
		std::cout << "The label magic is incorrect" << "\n";
		return 1;
	}
	return 0;
}

int check_labels(std::ifstream &fimg, std::ifstream &flabel, uint32_t &n, int len)
{
	uint32_t nImgs = 0, nLabels = 0;
	fimg.read(reinterpret_cast<char*>(&nImgs), len);
	if (!fimg)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

	flabel.read(reinterpret_cast<char*>(&nLabels), len);
	if (!flabel)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

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
	if (!fimg)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

	fimg.read(reinterpret_cast<char*>(&cols), len);
	if (!fimg)
	{
		std::cout << "Failed to read the stream" << std::endl;
		return 1;
	}

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

int load_mnist_info(DatasetOpts &opts, Info &o, std::string type)
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
	std::streampos pos = 0;
	int size = static_cast<int>(rows) * static_cast<int>(cols);
	o.reserve(n);
	// naive since, it expects all images to be the same size
	for (int i = 0; i < n; ++i)
	{
		flabel.read(&label, 1);
		if (!flabel)
		{
			std::cout << "Failed to read the stream" << std::endl;
			return 1;
		}

		pos = fimg.tellg();
		if (pos == std::streampos(-1))
		{
			std::cout << "Failed to read the stream" << std::endl;
			return 1;
		}

		o.push_back(std::make_pair(pos, label));
		pos += size;
		fimg.seekg(pos, std::ios::beg);
		if (!fimg)
		{
			std::cout << "Failed to read the stream" << std::endl;
			return 1;
		}
	}

	return 0;
}

int load_cifar10_batch_info(std::ifstream &stream, Info &o, uint32_t bs, uint32_t imgsz, uint32_t labelsz)
{
	char label;
	std::streampos pos = 0;

	for (uint32_t i = 0; i < bs; ++i)
	{
		stream.read(&label, labelsz);
		if (!stream)
		{
			std::cout << "Failed to read the stream" << std::endl;
			return 1;
		}
		pos = stream.tellg();
		if (pos == std::streampos(-1))
		{
			std::cout << "Failed to read the stream" << std::endl;
			return 1;
		}

		o.emplace_back(pos, label);
		pos += imgsz;;
		stream.seekg(pos, std::ios::beg);
		if (!stream)
		{
			std::cout << "Failed to move in the stream" << std::endl;
			return 1;
		}
	}
	return 0;
}

int load_cifar10_info(DatasetOpts &opts, Info &o, std::string type, uint32_t bs)
{
	/*
	* Think space
	* 
	* saving each path would take too much ram
	* -> Calculating path
	* 
	* switch case (dynamically calculating too slow)
	*	-> cache strings
	* int batch_idx = static_cast<int>(idx + 1000);
	*/
	
	int ret = 0;
	std::vector<std::string> fBatch = (type == "train") ? opts.fTrain : opts.fTest;
	o.reserve(fBatch.size() * bs);
	for (std::string &fBatch : fBatch)
	{
		std::ifstream s(fBatch, std::ios::in | std::ios::binary);
		ret = load_cifar10_batch_info(s, o, bs);
		if (ret != 0) { return ret; }
	}
	return 0;
}

std::optional<std::pair<cv::Mat, char>> load_mnist_img(std::string path, size_t i, const Info &d, uint32_t rows, uint32_t cols, int imgresz, uint32_t depth)
{
	std::ifstream fimg(path, std::ios::in | std::ios::binary);
	if (!fimg.is_open())
	{
		std::cout << "couldn't open the stream" << std::endl;
		return std::nullopt;
	}

	uint32_t imgsz = rows * cols * depth;
	std::streampos pos = d[i].first;
	char l = d[i].second;
	std::vector<char> buf(imgsz);

	fimg.seekg(pos, std::ios::beg);
	if (!fimg)
	{
		std::cout << "Couldn't read the stream" << std::endl;
		return std::nullopt;
	}

	fimg.read(buf.data(), buf.size());
	if (!fimg)
	{
		std::cout << "Couldn't read the stream" << std::endl;
		return std::nullopt;
	}

	cv::Mat img(rows, cols, CV_8UC1, buf.data());
	if (img.empty())
	{
		std::cout << "couldn't open the img" << std::endl;
		return std::nullopt;
	}
	if (imgresz != -1)
	{
		cv::resize(img, img, cv::Size(imgresz, imgresz));
	}

	return std::make_pair(img.clone(), l);
}

std::optional<cv::Mat> load_png_greyscale_img(std::string path, int imgresz)
{
	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		return std::nullopt;
	}
	if (imgresz != -1)
	{
		cv::resize(img, img, cv::Size(imgresz, imgresz));
	}
	return img;
}

torch::Tensor greyscale2Tensor(cv::Mat img, int imgsz, int div)
{
	torch::Tensor timg = torch::from_blob
	(
		img.data,
		{ 1, imgsz, imgsz },
		torch::kUInt8
	).to(torch::kFloat);
	if (div != -1)
	{
		timg.div_(div);
	}
	return timg;
}

torch::Tensor mat2Tensor(cv::Mat &img, int imgsz, int nc, int div)
{
	torch::Tensor timg = torch::from_blob(
		img.data,
		{ imgsz, imgsz, nc},
		torch::kUInt8
	).to(torch::kFloat);
	timg = timg.permute({ 2, 0, 1 });
	if (div != -1)
	{
		return timg.div_(div);
	}
	return timg;
}

std::optional<cv::Mat> Tensor2mat(torch::Tensor timg, int squeeze, std::pair<std::vector<double>, std::vector<double>> scale)
{
	timg = timg.detach().cpu();
	if (squeeze != -1) { timg.squeeze_(squeeze); }
	if (!scale.first.empty() && !scale.second.empty())
	{
		torch::Tensor tmean = torch::tensor(scale.first).view({ -1, 1, 1 });
		torch::Tensor tstdev = torch::tensor(scale.second).view({ -1, 1, 1 });
		timg.mul_(tstdev).add_(tmean)
			.mul_(255);
	}
	timg.clamp_(0, 255);
	timg = timg.to(torch::kUInt8);

	int nc = timg.size(0);
	int cv_maketype = CV_MAKETYPE(CV_8U, nc);

	int rows = timg.size(1), cols = timg.size(2);
	if (nc == 1) { timg.squeeze_(0); }

	cv::Mat img(rows, cols, cv_maketype, timg.data_ptr());
	if (img.empty()) { return std::nullopt; }
	return img.clone();
}
