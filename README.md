# CV application project (still in progress)

A simple ResNet classification project with cpp. This is my first cpp project, so be warned.

## Description

This project includes a custom training loop for ResNet and Lenet. The testing includes metrics
such as recall, precision and accuracy with a confusion matrix. In addition, the
inference supports png images (I made mine in paint). Inference also supports
feature map visualisations either by index or gradcam.

## Getting Started

### Dependencies

Here is a list of dependencies. However, some of them might not be required,
but no alternatives are tested.

* Windows 11
* Visual Studio 2022 (17.14.10) (recommended)
* nvidia rtx 2070 super (any card thats supports cuda 12.6 should be fine, if using gpu)

* msvc v143 for (x64/x86) (14.40-17.10) (recommended)
* ninja (1.12.1) (recommended)
* cmake (3.31.6) (required)
* mnist dataset (if you want to train a model by yourself, i donwloaded mine from kaggle)
* cifar10 dataset (same as before but from https://www.cs.toronto.edu/%7Ekriz/cifar.html)
* Image Watch for Visual Studio 2022 (you can visualise imgs when debugging)

* cuda 12.6 (could work without if using cpu)
* libtorch 2.8.0 DEBUG (required, release might work too on linux)
* opencv 4.9.0 (built from source with cmake and ninja)

### Installing

First you need to install the packages (and datasets?). In addition, you need to install the correct
versions of compilers builders and IDEs.

### Executing program

First you need to have a look at the settings.h file, where you need to change
the paths of the directories. In addition, have a look at the cmake files and
change the directories there. Lastly, if you haven't done yet so, you need to add
cuda, libtorch and opencv to path.

TODO fix this (easier to compile and run with vs 2022):

```
git clone https://github.com/jesseValkama/cv_project_cpp
```
```
cmake -S <src> . -B <out/build> -G Ninja -B -DCMAKE_CXX_COMPILER=cl --config Debug
```
```
cmake --build <build> J <nThreads>
```
```
./application_project train x test x inference x xai x model x
```
train: 0 || 1

test: 0 || 1

inference: 0 || 1

xai: -2 (skip), -1 (gradcam), 0-n feature map index

model: 1 (LeNet), 2 (ResNet)

dataset: 1 (mnist), 2 (cifar10)


## Help

If there is an issue with a nullptr when forward passing, you are likely using
the release version of libtorch on Windows on debug mode. 

This project has not been tested with a vm.

## Authors

Jesse Valkama 

## Acknowledgments

Credits to the following links, since they were helpful. Other more acknowledgements like small code snippets
from stackoverflow can be found in the .h files.
* [Cifar10 dataset](https://www.cs.toronto.edu/%7Ekriz/learning-features-2009-TR.pdf)
* [Mnist dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* [Libtorch tutorial code snippets](https://github.com/pytorch/examples/tree/main/cpp)
* [Tutorial videos for cpp](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)
* [Template for this readme](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)

## References

Credits to the following research papers.
* [Cifar10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
* [Gradcam](https://arxiv.org/abs/1610.02391)
* [LeNet & minst](https://arxiv.org/abs/1610.02391)
* [Resnet](https://arxiv.org/abs/1512.03385)