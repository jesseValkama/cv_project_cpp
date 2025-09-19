#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <opencv2/opencv.hpp>

#include <utility>

int mt_rand(const int min, const int max);
/*
* Generates a random number between min and max using the mersenne twister gen
* 
* Args:
*	min: the min value
*	max: the max value
* 
* Returns:
*	int: the random value
*/

bool apply_aug(const float prob = 0.5);
/*
* Determines whether to apply the augmentation
* 
* Args:
*	prob: the probability of applying the augmentation
* 
* Returns:
*	bool: whether to apply or not
*/

std::pair<int, int> rand_tl(const int x, const int y);
/*
* Generates a random top left point for an image
* 
* Args:
*	x: the max x value
*	y: the max y value
*	(How much space does the rest of the image 
*	need when adding the width and height)
* 
* Returns:
*	pair<int, int>: the top left point of the image
*/

namespace aug
{
	void crop(cv::Mat &img, const int pad = 4, const float prob = 0.5);
	/*
	* Crops the image by first applying padding and then cropping to the
	* original image size from a random point
	* 
	* Args:
	*	img: the image to be modified
	*	pad: the padding that is applied
	*	prob: the probability for the augmentation
	*/

	void mirror(cv::Mat &img, const int axel, const float prob = 0.5);
	/*
	* Mirrors the image either by x or y axel
	* 
	* Args:
	*	img: the image to be modified
	*	axel: the axel to be modified (0 is vertical, 1 is horizontal)
	*	prob: the probability for the augmentation
	*/
}

#endif