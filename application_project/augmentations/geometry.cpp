#include "geometry.h"

#include <opencv2/opencv.hpp>

#include <random>
#include <utility>

int mt_rand(const int min, const int max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(min, max);
	return dist(gen);
}

bool apply_aug(const float prob)
{
	int r = mt_rand(1, 100);
	return (static_cast<float>(r) / 100.0 <= prob) ? true : false;
}

std::pair<int, int> rand_tl(const int x, const int y)
{
	int rx = mt_rand(0, x), ry = mt_rand(0, y);
	return std::make_pair(rx, ry);
}

void aug::crop(cv::Mat &img, const int pad, const float prob)
{
	if (apply_aug(prob))
	{
		int w = img.cols, h = img.rows;
		cv::Mat bordered = img.clone();
		cv::copyMakeBorder(img, bordered, pad, pad, pad, pad, cv::BORDER_CONSTANT, { 0,0,0 });
		std::pair<int, int> tl = rand_tl(bordered.cols - w, bordered.rows - h);
		img = bordered(cv::Rect(tl.first, tl.second, w, h)).clone();
	}
}

void aug::mirror(cv::Mat &img, const int axel, const float prob)
{
	if (apply_aug(prob))
	{
		cv::flip(img, img, axel);
	}
}