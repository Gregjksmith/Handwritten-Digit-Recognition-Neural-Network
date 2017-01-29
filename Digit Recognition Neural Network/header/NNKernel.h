#pragma once

#include "../header/ClKernel.h"
#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <time.h>

#define NN_KENRNEL_SOURCE "kernels/NeuralNetwork.cl"

#define NN_INPUT_SIZE (28*28)
#define NN_LAYER_1_SIZE 300
#define NN_LAYER_2_SIZE 100
#define NN_OUTPUT_SIZE 10

#define NN_ACTIVATION_SIZE (NN_LAYER_1_SIZE + NN_LAYER_2_SIZE + NN_OUTPUT_SIZE)


#define NN_LAYER_1_WEIGHT_SIZE		(NN_INPUT_SIZE*NN_LAYER_1_SIZE)
#define NN_LAYER_1_BIAS_SIZE	NN_LAYER_1_SIZE
#define NN_LAYER_2_WEIGHT_SIZE		(NN_LAYER_1_SIZE*NN_LAYER_2_SIZE)
#define NN_LAYER_2_BIAS_SIZE	NN_LAYER_2_SIZE
#define NN_LAYER_OUTPUT_WEIGHT_SIZE		(NN_LAYER_2_SIZE*NN_OUTPUT_SIZE)
#define NN_LAYER_OUTPUT_BIAS_SIZE		NN_OUTPUT_SIZE

#define NN_WEIGHT_SIZE (NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE + NN_LAYER_OUTPUT_WEIGHT_SIZE + NN_LAYER_OUTPUT_BIAS_SIZE)

struct NNActivation
{
	float activationLayer1[NN_LAYER_1_SIZE];
	float activationLayer2[NN_LAYER_2_SIZE];
	float activationOutput[NN_OUTPUT_SIZE];
};

struct NNParams
{
	float layer1Weights[NN_LAYER_1_WEIGHT_SIZE];
	float layer1Bias[NN_LAYER_1_BIAS_SIZE];

	float layer2Weights[NN_LAYER_2_WEIGHT_SIZE];
	float layer2Bias[NN_LAYER_2_BIAS_SIZE];

	float layerOutputWeights[NN_LAYER_OUTPUT_WEIGHT_SIZE];
	float layerOutputBias[NN_LAYER_OUTPUT_BIAS_SIZE];
};

float getWeightLayer1(float* w, int inputIndex, int layer1Index);
float getBiasLayer1(float* w, int layer1Index);

float getWeightLayer2(float* w, int layer1Index, int layer2Index);
float getBiasLayer2(float* w,int layer2Index);

float getWeightLayerOutput(float* w, int layer2Index, int layerOutputIndex);
float getBiasLayerOutput(float* w, int layerOutputIndex);

using namespace cv;

#define STOCHASTIC_SAMPLING_SIZE 200
#define NUM_ITERS_EARLY_STOPPING 1
#define SAMPLING_ITERATIONS 5000
#define EARLY_STOPPING_COST 0.0000001f

/*
class NNKernel
Neural Network class used for digit recognition.

Contructor:
@param CLContext*
@param vector<Mat*> input image training set
@param vector<unsigned char> image label training set.

void train()
trains the neural network parameters using the training set.

unsigned char predict(Mat* image)
predicts the label of the image using the neural network.
@param Mat* input image

@return redicted label.
*/
class NNKernel : public ClKernel
{
public:
	/*CONSTRUCTOR*/
	NNKernel(ClContext* context, std::vector<Mat*> &inputImage, std::vector<unsigned char> &trainingLabels);
	/*DESTRUCTOR*/
	virtual ~NNKernel();
	
	/*
	void train
	trains the Neural Network parameters using the training images and training labels.
	*/
	virtual void train();

	/*
	float totalCost
	computes the total cost to the neural network with the training set.
	*/
	float totalCost();

	/*
	float test
	test the Neural Network accuracy on a test set.
	@return returns the error rate [0,1]
	*/
	float test(std::vector<Mat*> &inputImage, std::vector<unsigned char> &labels);

	/*
	unsigned char predict
	predicts the digit classification using the neural network.
	@return unsigned char, predicted digit [0,9]
	*/
	unsigned char predict(Mat* inputImage);

	void exportNNParams(char* filePath);
	void importNNParams(char* filePath);
	void exportReport(char* filePath);
	void exportReport(char* filePath, vector<Mat*> &testImages, vector<unsigned char> &testLabels);

private:

	/*BUFFERS AND VARIABLES*/
	float nnParams[NN_WEIGHT_SIZE];
	float activations[NN_ACTIVATION_SIZE];
	float activationDeltas[NN_ACTIVATION_SIZE];
	float gradient[NN_WEIGHT_SIZE];
	float learningParameter[NN_WEIGHT_SIZE];
	float cost;
	float learningFactor;

	float* inputImageVector;
	float* trainingLabelsVector;
	size_t numTrainingSample;

	/*
	void addNNKernelArg
	adds kernel arguments to each kernel function
	*/
	void addNNKernelArg();

	/*
	void createBuffers
	creates all the buffers necessary for the Neural Network
	@param vector<Mat*> : image training set.
	@param vector<unsigned char> : label training set.
	*/
	void createBuffers(std::vector<Mat*> &inputImage, std::vector<unsigned char> &trainingLabels);

	/*
	float* createImageVector
	vectorizes the image training set.
	@param vector<Mat*> : image training set

	@return float* : pointer to the images vector 
	*/
	float* createImageVector(std::vector<Mat*> &inputImage);

	/*
	float* createOutputVector
	vectorizes the output training labels.
	@param vector<unsigned char> : training labels

	@return float* : pointer to the labels vector
	*/
	float* createOutputVector(std::vector<unsigned char> &trainingLabels);

	/*
	void initNNParams
	initializes the neural network weights. The weights are sampled from a uniform distribution. 
	*/
	void initNNParams();

	/*
	void initGradientVector
	sets the gradient memory objects to zero.
	*/
	void initGradientVector();

	/*
	void setImageIndex
	sets the image index of the relevant kernels
	*/
	void setImageIndex(int index);

	/*functions that read memory object buffers into host memory*/
	void readNNParams();
	void readActivations();
	void readGradient();
	void readActivationDelta();
	void readCost();
	void readLearningParams();
	void readBuffers();

	/* 
	void clearBuffers
	clears the gradient buffer.
	*/
	void clearGradient();

	/*
	void clearCost
	clears the cost buffer.
	*/
	void clearCost();

	/*
	void clearLearningParameter
	clears the learning parameters used for adaptive weighted learning.
	*/
	void clearLearningParameter();
	
	/*
	void computeCost
	computes the NN cost from the activation buffer and the label buffer
	and adds it to the cost buffer.
	*/
	void computeCost();

	/*
	void normalizeGradient
	divides the gradient vector by the total number of training samples.
	(the cost is the average l2 norm)
	*/
	void normalizeGradient();

	/*
	void updateLearningParams
	update the adaptive learning factors using the vector-wise square of the gradient.
	*/
	void updateLearningParams();
	/*
	void updateWeightsRMS
	updates the NNweights of the Neural Network using the RMSProp algorithm.
	*/
	void updateWeightsRMS();

	/*
	void updateNNParams
	updates the NNweights of the Neural Network.
	*/
	void updateNNParams();

	/*
	float gradientInnerProduct
	computes the inner product of the gradient vector.
	*/
	float gradientInnerProduct();
	/*
	float learningParamInnerProduct
	computes the inner product of the learning param vector.
	*/
	float learningParamInnerProduct();

	/*
	void calculateActivationsLayer1
	computes the activations in layer 1 and places them in the activation buffer.
	*/
	void calculateActivationsLayer1();

	/*
	void calculateActivationsLayer2
	computes the activations in layer 2 and places them in the activation buffer.
	*/
	void calculateActivationsLayer2();

	/*
	void calculateActivationsLayerOutput
	computes the activations in the output layer and places them in the activation buffer.
	*/
	void calculateActivationsLayerOutput();

	/*
	void calculateActivationsDeltaLayer1
	computes the activations delta in layer 1 and places them in the activation delta buffer.
	*/
	void calculateActivationsDeltaLayer1();

	/*
	void calculateActivationsDeltaLayer2
	computes the activations delta in layer 2 and places them in the activation delta buffer.
	*/
	void calculateActivationsDeltaLayer2();

	/*
	void calculateActivationsDeltaLayerOutput
	computes the activations delta in the output layer and places them in the activation delta buffer.
	*/
	void calculateActivationsDeltaLayerOutput();

	/*
	void addGradientWeightLayer1
	computes the gradient of each weight in layer 1 and adds it to the gradient buffer
	*/
	void addGradientWeightLayer1();

	/*
	void addGradientWeightLayer2
	computes the gradient of each weight in layer 2 and adds it to the gradient buffer
	*/
	void addGradientWeightLayer2();


	/*
	void addGradientWeightLayerOutput
	computes the gradient of each weight in the output layer and adds it to the gradient buffer
	*/
	void addGradientWeightLayerOutput();

	/*
	void addGraidnetBiasLayer1
	computes the gradient of the biases in layer 1 and adds it to the gradient buffer.
	*/
	void addGradientBiasLayer1();

	/*
	void addGraidnetBiasLayer2
	computes the gradient of the biases in layer 2 and adds it to the gradient buffer.
	*/
	void addGradientBiasLayer2();

	/*
	void addGraidnetBiasLayerOutput
	computes the gradient of the biases in the output layer and adds it to the gradient buffer.
	*/
	void addGradientBiasLayerOutput();


	/*MEMORY OBJECTS*/
	cl_mem memobjInputVector;
	cl_mem memobjOutputTruthVector;
	cl_mem memobjActivationVector;
	cl_mem memobjNNParamsVector;
	cl_mem memobjActivationDeltaVector;
	cl_mem memobjGradientVector;
	cl_mem memobjCost;
	cl_mem memobjLearningParameter;

	/* REPORT VARIABLES*/
	time_t elapsedTime;
	std::vector<float> miniBatchCostHistory;

};