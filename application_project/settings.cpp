// todo: change the cpp file to a yaml file, this is just tmp

#include "settings.h"

DatasetOpts::DatasetOpts(DatasetTypes datasetType)
{

	MnistOpts mnistOpts;
	Cifar10Opts cifar10Opts;
	switch (datasetType)
	{
		case DatasetTypes::InitType:
			break;

		case DatasetTypes::MnistType:
			assign_from_mnist(mnistOpts);
			break;

		case DatasetTypes::Cifar10Type:
			assign_from_cifar10(cifar10Opts);
			break;
	}
}

void DatasetOpts::assign_from_mnist(MnistOpts &mnistOpts)
{
	fTrainImgs = mnistOpts.fTrainImgs;
	fTrainLabels = mnistOpts.fTrainLabels;
	fTestImgs = mnistOpts.fTestImgs;
	fTestLabels = mnistOpts.fTestLabels;

	inferenceDataPath = mnistOpts.inferenceDataPath;
	savepath = mnistOpts.savepath;
	inferenceModel = mnistOpts.inferenceModel;
	testModel = mnistOpts.testModel;
	workModel = mnistOpts.workModel;

	imgsz = mnistOpts.imgsz;
	imgresz = mnistOpts.imgresz;
	trainBS = mnistOpts.trainBS;
	valBS = mnistOpts.valBS;
	testBS = mnistOpts.testBS;
	numWorkers = mnistOpts.numWorkers;
	numOfClasses = mnistOpts.numOfClasses;
	numOfChannels = mnistOpts.numOfChannels;
	
	mean = mnistOpts.mean;
	stdev = mnistOpts.stdev;
}

void DatasetOpts::assign_from_cifar10(Cifar10Opts &cifar10Opts)
{
	trainBatch1 = cifar10Opts.trainBatch1;
	trainBatch2 = cifar10Opts.trainBatch2;
	trainBatch3 = cifar10Opts.trainBatch3;
	trainBatch4 = cifar10Opts.trainBatch4;
	trainBatch5 = cifar10Opts.trainBatch5;
	meta = cifar10Opts.meta;
	testBatch1 = cifar10Opts.testBatch1;
	fTrain = cifar10Opts.fTrain;
	fTest = cifar10Opts.fTest;

	inferenceDataPath = cifar10Opts.inferenceDataPath;
	savepath = cifar10Opts.savepath;
	inferenceModel = cifar10Opts.inferenceModel;
	testModel = cifar10Opts.testModel;
	workModel = cifar10Opts.workModel;

	imgsz = cifar10Opts.imgsz;
	imgresz = cifar10Opts.imgresz;
	trainBS = cifar10Opts.trainBS;
	valBS = cifar10Opts.valBS;
	testBS = cifar10Opts.testBS;
	numWorkers = cifar10Opts.numWorkers;
	numOfClasses = cifar10Opts.numOfClasses;
	numOfChannels = cifar10Opts.numOfChannels;
	
	mean = cifar10Opts.mean;
	stdev = cifar10Opts.stdev;
}

Settings::Settings(DatasetTypes datasetType)
{
	DatasetOpts datasetOpts(datasetType);
	mnistOpts = datasetOpts;
}
