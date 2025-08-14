#include "tmp.h"

#include <stdint.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

/*
* I have little experience with lowlevel reading files, so i am trying it out here first to get something working
* Credit: https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c 
* Though I have done some of my own changes
*/

uint32_t swap_endian(uint32_t val)
{
	/*
	*/

	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

int vis_mnist(const char *flabelname, const char *fimgname)
{
	/*
	* How the files work (I think?)
	* 
	* Seriously do you guys need to understand
	* everything about files if you work with cpp
	* 
	* idx3 = imgs: n * x * y
	* idx1 = labels: n
	* 
	* Headers (big-endian):
	* contains the metadata and the
	* data to identify it (magic?)
	*/

	std::ifstream flabel(flabelname, std::ios::in | std::ios::binary);
	std::ifstream fimg(fimgname, std::ios::in | std::ios::binary);

	if (!(flabel.is_open() && fimg.is_open()))
	{
		std::cout << "Could not open the stream" << "\n";
		return 1;
	}

	uint32_t magic, numImgs, numLabels, rows, cols;
	uint32_t labelMagicNum = 2049, imgMagicNum = 2051; // variables used so i can understand what is happening here
	
	flabel.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	fimg.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != imgMagicNum)
	{
		std::cout << "Incorrect img magic" << "\n";
	}
	if (magic != labelMagicNum)
	{
		std::cout << "Incorrect label magic" << "\n";
	}
	

	fimg.read(reinterpret_cast<char*>(&numImgs), 4);
	numImgs = swap_endian(numImgs);
	flabel.read(reinterpret_cast<char*>(&numLabels), 4);
	numLabels = swap_endian(numLabels);
	if (numImgs != numLabels)
	{
		std::cout << "n of labels doesn't correspond the the n of imgs" << "\n";
		return 2;
	}

	fimg.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	fimg.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);
	// the original mnist dataset used 32 x 32, which was changed to 28x28
	if (!(rows == 28 && cols == 28))
	{
		std::cout << "Weird img size" << "\n";
		return 3;
	}

	char label = 0;
	std::string sLabel;
	std::unique_ptr<char*> pixels = std::make_unique<char*>();

	cv::namedWindow("mnist", cv::WINDOW_AUTOSIZE);

	fimg.read(*pixels, rows * cols);
	flabel.read(&label, 1);

	sLabel = std::to_string(int(label));
	cv::Mat img(rows, cols, CV_8UC1, *pixels);
	cv::resize(img, img, cv::Size(100, 100));
	cv::imshow(sLabel, img);
	cv::waitKey(0);

	return 0;
}