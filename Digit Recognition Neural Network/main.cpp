
/*
DIGIT RECOGNITION USING CONVOLUTIONAL NEURAL NETWORKS

Implementation of "Gradient-Based Learning Applied to Document Recognition"
	LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

%%Architecture%%

-input : handwritten image

-c1 : convolutional layer


%%Training Set%%

%%Learning%%

*/

#include "header\ConvolutionalNeuralNetwork.h"
#include "CL\cl.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <opencv2\opencv.hpp>
#include "header\ClContext.h"
#include "header\CNNKernel.h"
#include "header\NNKernel.h"

#define NUM_TRAINING_IMAGES 60000
#define TRAINING_IMAGES_WIDTH 28
#define TRAINING_IMAGES_HEIGHT 28
#define TRAINING_SET_IMAGE_PATH "../Training Set/train-images.idx3-ubyte"
#define TRAINING_SET_LABEL_PATH "../Training Set/train-labels.idx1-ubyte"

#define NUM_TEST_IMAGES 10000
#define TEST_SET_IMAGE_PATH "../Training Set/t10k-images.idx3-ubyte"
#define TEST_SET_LABEL_PATH "../Training Set/t10k-labels.idx1-ubyte"

/*
void loadTrainingSet : loads the training set of handwritten digits and places the images in 'trainingImages' and the lables in 'trainingLabels'.
the training set comes from the MNIST databse and contains 60000 uniqe 28x28 handwritten digits.
@param imageFilePath : specifies the filepath of the image training set.
@param labelFilePath : specifices the filepath of the label training set.
@param trainingSetSize : specifies the size of the training set.
@param trainingImages : vector used to contain the training images.
@param trainingLabels : vector used to contain the training labels.
*/
void loadTrainingSet(char* imageFilePath, char* labelFilePath, int trainingSetSize, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels);

#define NN_PARAMS_EXPORT_PATH "../exports/exportNNParams.bin"
#define REPORT_PATH "../exports/NNreport.txt"

int main()
{
	vector<Mat*> trainingImages;
	vector<unsigned char> trainingLabels;
	vector<Mat*> testImages;
	vector<unsigned char> testLabels;

	loadTrainingSet(TRAINING_SET_IMAGE_PATH, TRAINING_SET_LABEL_PATH, NUM_TRAINING_IMAGES, trainingImages, trainingLabels);
	loadTrainingSet(TEST_SET_IMAGE_PATH, TEST_SET_LABEL_PATH, NUM_TEST_IMAGES, testImages, testLabels);

	ClContext* clContext = new ClContext();
	NNKernel* nn = new NNKernel(clContext, trainingImages, trainingLabels);
	nn->train();
	nn->exportNNParams(NN_PARAMS_EXPORT_PATH);
	nn->exportReport(REPORT_PATH);


	float testErrorRate = nn->test(testImages, testLabels);

	
}


void loadTrainingSet(char* imageFilePath, char* labelFilePath, int trainingSetSize, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels)
{
	fstream imageFile;
	fstream labelFile;

	imageFile.open(imageFilePath, std::fstream::in | std::fstream::binary);
	labelFile.open(labelFilePath, std::fstream::in | std::fstream::binary);

	int imageHeader1, imageHeader2, imageHeader3, imageHeader4;
	int labelHeader1, labelHeader2;

	imageFile.read((char*)&imageHeader1, 4);
	imageFile.read((char*)&imageHeader2, 4);
	imageFile.read((char*)&imageHeader3, 4);
	imageFile.read((char*)&imageHeader4, 4);

	labelFile.read((char*)&labelHeader1, 4);
	labelFile.read((char*)&labelHeader2, 4);

	unsigned char pixel;
	unsigned char label;
	Mat* image;

	for (int numImages = 0; numImages < trainingSetSize; numImages++)
	{
		image = new Mat(TRAINING_IMAGES_WIDTH, TRAINING_IMAGES_HEIGHT, CV_32F);
		for (int y = 0; y < TRAINING_IMAGES_WIDTH; y++)
		{
			for (int x = 0; x < TRAINING_IMAGES_HEIGHT; x++)
			{
				imageFile.read((char*)&pixel, 1);
				image->at<float>(y, x) = ((float)pixel)/255.0;
			}
		}
		trainingImages.push_back(image);

		labelFile.read((char*)&label, 1);
		trainingLabels.push_back(label);
	}

	imageFile.close();
	labelFile.close();

}