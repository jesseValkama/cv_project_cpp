# CV application project (still in progress)

A simple LeNet MNIST classification project with cpp (to "prove" my cpp skills).

## Description

This project includes a custom training loop for lenet. The testing includes metrics
such as recall, precision and accuracy with a confusion matrix. In addition, the
inference supports png images (I made mine in paint). Inference also supports
feature map visualisations.

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
* Image Watch for Visual Studio 2022 (you can visualise imgs when debugging)

* cuda 12.6 (could work without if using cpu)
* libtorch 2.8.0 DEBUG (required, release might work too on linux)
* opencv 4.9.0 (built from source with cmake and ninja)

### Installing

First you need to install the packages (and dataset?). In addition, you need to install the correct
versions of compilers builders and IDEs.

### Executing program

First you need to have a look at the settings.h file, where you need to change
the paths of the directories. In addition, have a look at the cmake files and
change the directories there. Lastly, if you haven't done yet so, you need to add
cuda, libtorch and opencv to path.

TODO: (I use VS 2022 so these are complex):

```
cmake -S <src> . -B <out/build> -G Ninja -B -DCMAKE_CXX_COMPILER=cl --config Debug
```
```
cmake --build <build> J <nThreads>
```
```
./application_project train x test x inference x
```
where x is either 0 or 1.

## Help

If there is an issue with a nullptr when forward passing, you are likely using
the release version of libtorch on Windows on debug mode. 

This project has not been tested with a vm.

## Authors

Jesse Valkama 

## Acknowledgments

Credits to the following links, since they were helpful. Other more acknowledgements like small code snippets
from stackoverflow can be found in the .h files.
* [Mnist dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* [Libtorch tutorial code snippets](https://github.com/pytorch/examples/tree/main/cpp)
* [Tutorial videos for cpp](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)
* [Template for this readme](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)