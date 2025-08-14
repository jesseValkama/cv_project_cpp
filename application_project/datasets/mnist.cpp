#include "mnist.h"

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdint.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>

// TOOD: install torchvision to get most basic augmentations working

Example MnistDataset::get(size_t i)
{
	std::string path = mnistOpts.datasetPath + data[i].first;
	cv::Mat img = cv::imread(path);
	assert(!img.empty());

	cv::resize(img, img, cv::Size(mnistOpts.imgSize, mnistOpts.imgSize));
	std::vector<cv::Mat> channels(3);
	cv::split(img, channels);

	auto B = torch::from_blob
	(
		channels[0].ptr(),
		{ mnistOpts.imgSize, mnistOpts.imgSize },
		torch::kUInt8
	);
	auto G = torch::from_blob
	(
		channels[1].ptr(),
		{ mnistOpts.imgSize, mnistOpts.imgSize },
		torch::kUInt8
	);
	auto R = torch::from_blob
	(
		channels[2].ptr(),
		{ mnistOpts.imgSize, mnistOpts.imgSize },
		torch::kUInt8
	);

	auto tdata = torch::cat({ R, G, B })
		.view({ 3, mnistOpts.imgSize, mnistOpts.imgSize })
		.to(torch::kFloat);
	auto tlabel = torch::tensor(data[i].second, torch::kLong);
	return { tdata, tlabel };
}

torch::optional<size_t> MnistDataset::size() const
{
	return data.size();
}

std::pair<Data, Data> readInfo(void)
{
	std::random_device randDev;
	std::mt19937 mersenneTwisterGenerator(randDev());
	Data train, val, test;
	
	std::ifstream stream(mnistOpts.infoFilePath, std::ios::binary);
	assert(stream.is_open());
	
	int64_t label;
	std::string path, type;
	while (true)
	{
		stream >> path >> label >> type;
		
		// I wish c++ could use str in switch case
		if (type == "train")
		{
			train.push_back(std::make_pair(path, label));
		}
		else if (type == "val")
		{
			val.push_back(std::make_pair(path, label));
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
	
	// TODO: add possibility disable shuffling
	std::shuffle(train.begin(), train.end(), mersenneTwisterGenerator);
	std::shuffle(val.begin(), val.end(), mersenneTwisterGenerator);
	std::shuffle(test.begin(), test.end(), mersenneTwisterGenerator);
	return std::make_pair(train, test); //change this to include val too 
}