# CV application project

A simple ResNet/Cifar10 classification project with cpp. This is my first cpp project, older code might be rough. Some of this is in process of being rewritten (if/when it becomes a problem)

## Description

This project includes a custom training loop for ResNet and Lenet. The testing includes metrics
such as recall, precision and accuracy with a confusion matrix. In addition, the
inference supports png images (I made mine in paint for mnist, use your own for cifar10). Inference also supports
feature map visualisations either by index or gradcam.

## Getting Started

### Dependencies

I use two systems, a laptop (Linux Mint) and main desktop (Windows 11).

#### General
* nvidia rtx 2070 super (cuda 12.6)
* mnist dataset (if you want to train a model by yourself, i donwloaded mine from kaggle)
* cifar10 dataset (same as before but from https://www.cs.toronto.edu/%7Ekriz/cifar.html)

#### Linux Mint 22.3 (recommended)
* VS Code (latest)
* g++ (13.3.0)
* clang (18.1.3) (might start using with g++ to make sense of errmsgs)
* valgrind (3.22.0)
* ninja (1.11.1)
* cmake (3.28.3)
* opencv
* libtorch

#### Windows 11 (i am hoping to move my main pc away from this soon)
* Visual Studio 2022 (17.14.10)
* msvc v143 for (x64/x86) (14.40-17.10)
* cmake (3.31.6) (required)
* ninja (1.12.1)
* Image Watch for Visual Studio 2022 (you can visualise imgs when debugging)
* libtorch (2.8.0 DEBUG)
* opencv (4.9.0, built from source with cmake and ninja)
* cuda (12.6)

### Installing

First you need to install the packages (and datasets?). In addition, you need to install the correct
versions of compilers, build tools, and IDEs.

### Executing program

The paths are not constructed properly yet. Hence you need to change them in settings.h, settings.yaml, and cmake/{linux/windows}/paths.cmake. In addition, for Windows, some paths like Cuda, LibTorch, and OpenCV might need to be added to path.

```
git clone https://github.com/jesseValkama/cv_project_cpp
```
I use the following on Linux, on Windows CTRL + S on CMakeLists.txt for configurating the program
```
chmod +x gen.sh
```
```
./gen.sh
```
I use the following on Linux, on Windows CTRL + B for building the program
```
cmake --build build -j n
```
where, n is the number of jobs in parallel

I use the following on Linux, on Windows f5 and set up the args in .json file for running the program
```
./build/application_project/application_project args
```
where, args are:

train: 0 || 1

test: 0 || 1

inference: 0 || 1

xai: -2 (skip), -1 (gradcam), 0-n feature map index

model: 1 (LeNet), 2 (ResNet)

dataset: 1 (mnist), 2 (cifar10)

to use the database:
```
sqlite3 name.db
.schema
```
```
SELECT * FROM experiments JOIN metrics ON experiments.id = metrics.metrics_id JOIN config ON experiments.id = config.config_id;
```

## Help

If there is an issue with a nullptr when forward passing, you are likely using
the release version of libtorch on Windows. Please use the debug version. (This would never happen to me.)

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