#include "tmp.h"

#include <stdint.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include "loader_funcs.h"

/*
* I have little experience with lowlevel reading files, so i am trying it out here first to get something working
* Credit: https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c 
* Though I have done some of my own changes
*/

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

	std::ifstream fimg(fimgname, std::ios::in | std::ios::binary);
	std::ifstream flabel(flabelname, std::ios::in | std::ios::binary);

	if (!(flabel.is_open() && fimg.is_open()))
	{
		std::cout << "Could not open the stream" << "\n";
		return 1;
	}

	uint32_t magic, numImgs, numLabels, rows, cols;
	uint32_t labelMagicNum = 2049, imgMagicNum = 2051; // variables used so i can understand what is happening here
	
	flabel.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != labelMagicNum)
	{
		std::cout << "Incorrect label magic" << "\n";
		return 2;
	}
	fimg.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != imgMagicNum)
	{
		std::cout << "Incorrect img magic" << "\n";
		return 2;
	}

	fimg.read(reinterpret_cast<char*>(&numImgs), 4);
	numImgs = swap_endian(numImgs);
	flabel.read(reinterpret_cast<char*>(&numLabels), 4);
	numLabels = swap_endian(numLabels);
	if (numImgs != numLabels)
	{
		std::cout << "n of labels doesn't correspond the the n of imgs" << "\n";
		return 3;
	}

	fimg.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	fimg.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);
	// the original mnist dataset used 32 x 32, which was changed to 28x28
	if (!(rows == 28 && cols == 28))
	{
		std::cout << "Weird img size" << "\n";
		return 4;
	}

	char label;
	std::vector<char> buffer(rows * cols);
	int key = 0;
	cv::namedWindow("mnist", cv::WINDOW_AUTOSIZE);
	
	for (int i = 0; i < numImgs; ++i)
	{
		fimg.read(buffer.data(), rows * cols);
		flabel.read(&label, 1);

		cv::Mat img(rows, cols, CV_8UC1, buffer.data());
		cv::resize(img, img, cv::Size(100, 100));
		cv::imshow("mnist", img);
		key = cv::waitKey(0);
		if (key == 27)
		{
			std::cout << "Stopping" << "\n";
			break;
		}
	}
	
	return 0;
}
