#include "loader_funcs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <stdint.h>
#include <string>
#include <tuple>
#include <vector>

int load_mnist(std::string path, int64_t label, int imgSize, Example &o)
{
	cv::Mat img = cv::imread(path);
	if (img.empty())
	{
		std::cout << "unable to load an image" << std::endl;
		return 1;
	}

	cv::resize(img, img, cv::Size(imgSize, imgSize));
	std::vector<cv::Mat> channels(3);
	cv::split(img, channels);

	auto B = torch::from_blob
	(
		channels[0].ptr(),
		{ imgSize, imgSize },
		torch::kUInt8
	);
	auto G = torch::from_blob
	(
		channels[1].ptr(),
		{ imgSize, imgSize },
		torch::kUInt8
	);
	auto R = torch::from_blob
	(
		channels[2].ptr(),
		{ imgSize, imgSize },
		torch::kUInt8
	);

	auto tdata = torch::cat({ R, G, B })
		.view({ 3, imgSize, imgSize })
		.to(torch::kFloat);
	auto tlabel = torch::tensor(label, torch::kLong);
	o = { tdata, tlabel };

	return 0;
}

int load_mnist_info(std::string fname, uint8_t trainProb, std::tuple<Data, Data, Data> &o)
{
	assert(trainProb <= 100);
	Data train, val, test;
	
	std::ifstream stream(fname, std::ios::binary); 
	if (!stream.is_open())
	{
		std::cout << "Could not open file, like due to inproper filename" << std::endl;
		return 1;
	}
	
	int64_t label;
	std::string path, type;
	while (true)
	{
		stream >> path >> label >> type;
		
		// MNIST dataset has train and val already predefined, so i use those
		// however, there is no val predefined, so it is predefined at random
		if (type == "train")
		{
			if (std::rand() % 101 >= trainProb)
			{
				train.push_back(std::make_pair(path, label));
			}
			else
			{
				val.push_back(std::make_pair(path, label));
			}
		}
		else if (type == "test")
		{
			test.push_back(std::make_pair(path, label));
		}
		if (stream.eof())
		{
			break;
		}
	}
	
	o = std::make_tuple(train, val, test);
	return 0;
}