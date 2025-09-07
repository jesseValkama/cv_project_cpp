#include "loader_funcs.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <optional>
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
		if (pos == std::streampos(-1)) // should be correct, geeksforgeeks is not great
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

std::optional<std::pair<cv::Mat, char>> load_mnist_img(std::string path, size_t i, const Info &d, uint32_t rows, uint32_t cols, int imgresz)
{
	std::ifstream fimg(path, std::ios::in | std::ios::binary);
	if (!fimg.is_open())
	{
		std::cout << "couldn't open the stream" << std::endl;
		return std::nullopt;
	}

	uint32_t imgsz = rows * cols;
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

std::optional<cv::Mat> Tensor2greyscale(torch::Tensor timg, bool squeeze, std::pair<float, float> scale)
{
	torch::Tensor dbg = timg.detach().cpu();
	if (squeeze) { dbg.squeeze_(); }

	if (scale.first != -1.0 && scale.second != -1.0)
	{
		dbg = dbg.mul(scale.first).add(scale.second).mul(255).clamp(0,255);
	}
	dbg = dbg.to(torch::kUInt8);

	cv::Mat img(dbg.size(0), dbg.size(1), CV_8UC1, dbg.data_ptr());
	if (img.empty()) { return std::nullopt; }

	return img.clone();
}
