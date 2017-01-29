#include "../header/NNKernel.h" 

NNKernel::NNKernel(ClContext* context, std::vector<Mat*> &inputImage, std::vector<unsigned char> &trainingLabels) : ClKernel(NN_KENRNEL_SOURCE, context)
{
	cl_int ret;
	learningFactor = 1.0;
	createKernel("activationLayer1");
	createKernel("activationLayer2");
	createKernel("activationLayerOutput");

	createKernel("activationLayer1Delta");
	createKernel("activationLayer2Delta");
	createKernel("activationOutputDelta");
	
	createKernel("addGradientWeightLayer1");
	createKernel("addGradientWeightLayer2");
	createKernel("addGradientWeightLayerOutput");

	createKernel("addGradientBiasLayer1");
	createKernel("addGradientBiasLayer2");
	createKernel("addGradientBiasLayerOutput");

	createKernel("updateNNparams");
	createKernel("cost");
	createKernel("normalizeGradient");

	createKernel("updateLearningParams");
	createKernel("updateNNParamsRMS");

	createBuffers(inputImage, trainingLabels);
	addNNKernelArg();
	readBuffers();
}

NNKernel::~NNKernel()
{
	clReleaseMemObject(memobjInputVector);
	clReleaseMemObject(memobjOutputTruthVector);
	clReleaseMemObject(memobjActivationVector);
	clReleaseMemObject(memobjNNParamsVector);
	clReleaseMemObject(memobjActivationDeltaVector);
	clReleaseMemObject(memobjGradientVector);
	clReleaseMemObject(memobjCost);

	delete inputImageVector;
	delete trainingLabelsVector;
}

void NNKernel::train()
{
	elapsedTime = time(0);
	srand(time(NULL));

	int trainingSamples[STOCHASTIC_SAMPLING_SIZE];
	clearLearningParameter();

	for (int stochSampIndex = 0; stochSampIndex < SAMPLING_ITERATIONS; stochSampIndex++)
	{
		for (int tsIndex = 0; tsIndex < STOCHASTIC_SAMPLING_SIZE; tsIndex++)
		{
			float r =  ((float)numTrainingSample - 1.0f)*((float)rand() / (float)RAND_MAX);
			trainingSamples[tsIndex] = (int)floor(r);
		}

		for (int iter = 0; iter < NUM_ITERS_EARLY_STOPPING; iter++)
		{
			clearGradient();
			clearCost();
			for (int i = 0; i < STOCHASTIC_SAMPLING_SIZE; i++)
			{
				setImageIndex(trainingSamples[i]);
				calculateActivationsLayer1();
				calculateActivationsLayer2();
				calculateActivationsLayerOutput();
				calculateActivationsDeltaLayerOutput();
				calculateActivationsDeltaLayer2();
				calculateActivationsDeltaLayer1();
				addGradientBiasLayerOutput();
				addGradientWeightLayerOutput();
				addGradientBiasLayer2();
				addGradientWeightLayer2();
				addGradientBiasLayer1();
				addGradientWeightLayer1();
				computeCost();
			}
			normalizeGradient();
			readCost();
			cost = cost / (float)STOCHASTIC_SAMPLING_SIZE;

			updateNNParams();

			miniBatchCostHistory.push_back(cost);

		}
	}
	elapsedTime = time(0) - elapsedTime;
}

float NNKernel::totalCost()
{
	clearCost();
	for (int i = 0; i < numTrainingSample; i++)
	{
		setImageIndex(i);
		calculateActivationsLayer1();
		calculateActivationsLayer2();
		calculateActivationsLayerOutput();
		computeCost();
	}
	readCost();
	cost = cost / (float)numTrainingSample;
	return cost;
}

float NNKernel::test(std::vector<Mat*> &inputImage, std::vector<unsigned char> &labels)
{
	int numCorrect = 0;
	for (int i = 0; i < inputImage.size(); i++)
	{
		unsigned char p = predict(inputImage[i]);
		if (p == labels[i])
		{
			numCorrect++;
		}
	}

	float correctRate = (float)numCorrect / (float)inputImage.size();
	return (1.0f - correctRate);
}

unsigned char NNKernel::predict(Mat* inputImage)
{
	float imVec[NN_INPUT_SIZE];
	int cols = inputImage->cols;
	int rows = inputImage->rows;
	int index = 0;
	for (int y = 0; y < cols; y++)
	{
		for (int x = 0; x < rows; x++)
		{
			imVec[index] = inputImage->at<float>(x, y);
			index++;
		}
	}

	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, memobjInputVector, CL_TRUE, 0, NN_INPUT_SIZE*sizeof(float), imVec, 0, NULL, NULL);
	int sampleIndex = 0;
	addKernelArg(0, 1, sizeof(cl_int), (void*)&sampleIndex);
	clFinish(__context->commandQueue);


	calculateActivationsLayer1();
	calculateActivationsLayer2();
	calculateActivationsLayerOutput();

	readActivations();

	float maxOutActivation = activations[NN_LAYER_1_SIZE + NN_LAYER_2_SIZE];
	unsigned char maxOutActivationIndex = 0;
	for (int i = 0; i < NN_OUTPUT_SIZE; i++)
	{
		float a = activations[i + NN_LAYER_1_SIZE + NN_LAYER_2_SIZE];
		if (a > maxOutActivation)
		{
			maxOutActivation = a;
			maxOutActivationIndex = i;
		}
	}
	return maxOutActivationIndex;
}

float* NNKernel::createImageVector(std::vector<Mat*> &inputImage)
{
	int rows = inputImage[0]->rows;
	int cols = inputImage[0]->cols;
	float* imVec = new float[inputImage.size()*rows*cols];
	int imVecIndex = 0;
	for (int imageIndex = 0; imageIndex < inputImage.size(); imageIndex++)
	{
		for (int y = 0; y < cols; y++)
		{
			for (int x = 0; x < rows; x++)
			{
				imVec[imVecIndex] = inputImage[imageIndex]->at<float>(x, y);
				imVecIndex++;
			}
		}
	}
	return imVec;
}

float* NNKernel::createOutputVector(std::vector<unsigned char> &trainingLabels)
{
	float* outVec = new float[NN_OUTPUT_SIZE*trainingLabels.size()];
	int outVecIndex = 0;
	for (int outputIndex = 0; outputIndex < trainingLabels.size(); outputIndex++)
	{
		for (unsigned char i = 0; i < NN_OUTPUT_SIZE; i++)
		{
			if (i == trainingLabels[outputIndex])
				outVec[outVecIndex] = 1.0;
			else
				outVec[outVecIndex] = 0.0;

			outVecIndex++;
		}

	}
	return outVec;
}

void NNKernel::createBuffers(std::vector<Mat*> &inputImage, std::vector<unsigned char> &trainingLabels)
{
	cl_int ret;

	size_t inputImageWidth = inputImage[0]->rows;
	size_t inputImageHeight = inputImage[0]->cols;

	size_t inputBufferSize = inputImageWidth*inputImageHeight*inputImage.size()*sizeof(float);
	size_t outputVectorSize = NN_OUTPUT_SIZE*trainingLabels.size()*sizeof(float);
	size_t activationSize = NN_ACTIVATION_SIZE*sizeof(float);
	size_t nnParamSize = NN_WEIGHT_SIZE*sizeof(float);

	numTrainingSample = inputImage.size();
	float* inputImageVec = createImageVector(inputImage);
	float* outputImageVec = createOutputVector(trainingLabels);
	initNNParams();

	memobjInputVector = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, inputBufferSize, NULL, &ret);
	memobjOutputTruthVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, outputVectorSize, NULL, &ret);
	memobjActivationVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, NULL, &ret);
	memobjNNParamsVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, nnParamSize, NULL, &ret);
	memobjActivationDeltaVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, activationSize, NULL, &ret);
	memobjGradientVector = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, nnParamSize, NULL, &ret);
	memobjCost = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
	memobjLearningParameter = clCreateBuffer(__context->context, CL_MEM_READ_WRITE, nnParamSize, NULL, &ret);

	ret = clEnqueueWriteBuffer(__context->commandQueue, memobjInputVector, CL_TRUE, 0, inputBufferSize, inputImageVec, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(__context->commandQueue, memobjOutputTruthVector, CL_TRUE, 0, outputVectorSize, outputImageVec, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(__context->commandQueue, memobjNNParamsVector, CL_TRUE, 0, nnParamSize, nnParams, 0, NULL, NULL);
	float costBuf = 0;
	ret = clEnqueueWriteBuffer(__context->commandQueue, memobjCost, CL_TRUE, 0, sizeof(float), &cost, 0, NULL, NULL);

	initGradientVector();
	clearLearningParameter();

	inputImageVector = inputImageVec;
	trainingLabelsVector = outputImageVec;
}

void NNKernel::initNNParams()
{
	srand(time(NULL));

	float randRange = sqrt(6.0f / (float)(NN_INPUT_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_1_WEIGHT_SIZE; i++)
	{
		((NNParams*)nnParams)->layer1Weights[i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	randRange = sqrt(6.0f / (float)(NN_LAYER_2_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_2_WEIGHT_SIZE; i++)
	{
		((NNParams*)nnParams)->layer2Weights[i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	randRange = sqrt(6.0f / (float)(NN_OUTPUT_SIZE + NN_LAYER_1_SIZE));
	for (int i = 0; i < NN_LAYER_OUTPUT_WEIGHT_SIZE; i++)
	{
		((NNParams*)nnParams)->layerOutputWeights[i] = randRange*(2.0*((float)rand() / (float)RAND_MAX) - 1.0);
	}

	for (int i = 0; i < NN_LAYER_1_BIAS_SIZE; i++)
	{
		((NNParams*)nnParams)->layer1Bias[i] = 0.0;
	}

	for (int i = 0; i < NN_LAYER_2_BIAS_SIZE; i++)
	{
		((NNParams*)nnParams)->layer2Bias[i] = 0.0;
	}

	for (int i = 0; i < NN_LAYER_OUTPUT_BIAS_SIZE; i++)
	{
		((NNParams*)nnParams)->layerOutputBias[i] = 0.0;
	}
}

void NNKernel::initGradientVector()
{
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		gradient[i] = 0.0;
	}
	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), gradient, 0, NULL, NULL);
}

void NNKernel::addNNKernelArg()
{
	cl_int numSampleTemp = 0;

	/* ACTIVATION LAYER 1*/
	addKernelArg(0, 0, sizeof(cl_mem), (void*)&memobjInputVector);
	addKernelArg(0, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(0, 2, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(0, 3, sizeof(cl_mem), (void*)&memobjNNParamsVector);

	/* ACTIVATION LAYER 2*/
	addKernelArg(1, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(1, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);

	/* ACTIVATION LAYER OUTPUT */
	addKernelArg(2, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(2, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);



	/* ACTIVATION DELTA LAYER 1*/
	addKernelArg(3, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(3, 1, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(3, 2, sizeof(cl_mem), (void*)&memobjNNParamsVector);


	/* ACTIVATION DELTA LAYER 2*/
	addKernelArg(4, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(4, 1, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(4, 2, sizeof(cl_mem), (void*)&memobjNNParamsVector);

	/* ACTIVATION DELTA LAYER OUTPUT*/
	addKernelArg(5, 0, sizeof(cl_mem), (void*)&memobjOutputTruthVector);
	addKernelArg(5, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(5, 2, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(5, 3, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);



	/* GRADIENT WEIGHT LAYER 1*/
	addKernelArg(6, 0, sizeof(cl_mem), (void*)&memobjInputVector);
	addKernelArg(6, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(6, 2, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(6, 3, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(6, 4, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(6, 5, sizeof(cl_mem), (void*)&memobjGradientVector);


	/* GRADIENT WEIGHT LAYER 2*/
	addKernelArg(7, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(7, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(7, 2, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(7, 3, sizeof(cl_mem), (void*)&memobjGradientVector);


	/* GRADIENT WEIGHT LAYER OUTPUT*/
	addKernelArg(8, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(8, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(8, 2, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(8, 3, sizeof(cl_mem), (void*)&memobjGradientVector);


	/* GRADIENT BIAS LAYER 1*/
	addKernelArg(9, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(9, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(9, 2, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(9, 3, sizeof(cl_mem), (void*)&memobjGradientVector);

	/* GRADIENT BIAS LAYER 2*/
	addKernelArg(10, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(10, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(10, 2, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(10, 3, sizeof(cl_mem), (void*)&memobjGradientVector);

	/* GRADIENT BIAS LAYER OUTPUT*/
	addKernelArg(11, 0, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(11, 1, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(11, 2, sizeof(cl_mem), (void*)&memobjActivationDeltaVector);
	addKernelArg(11, 3, sizeof(cl_mem), (void*)&memobjGradientVector);

	/*UPDATE VECTOR*/
	addKernelArg(12, 0, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(12, 1, sizeof(cl_mem), (void*)&memobjGradientVector);
	//addKernelArg(12, 2, sizeof(cl_float), (void*)&learningFactor);

	/*COST FUNCTIOn*/
	addKernelArg(13, 0, sizeof(cl_mem), (void*)&memobjOutputTruthVector);
	addKernelArg(13, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(13, 2, sizeof(cl_mem), (void*)&memobjActivationVector);
	addKernelArg(13, 3, sizeof(cl_mem), (void*)&memobjCost);

	/*NORMALIZE GRADIENT*/
	addKernelArg(14, 0, sizeof(cl_mem), (void*)&memobjGradientVector);
	int trainingSample = STOCHASTIC_SAMPLING_SIZE;
	addKernelArg(14, 1, sizeof(cl_int), (void*)&trainingSample);

	/*UPDATE LEARNING PARAMS*/
	addKernelArg(15, 0, sizeof(cl_mem), (void*)&memobjLearningParameter);
	addKernelArg(15, 1, sizeof(cl_mem), (void*)&memobjGradientVector);

	/*UPDATE WEIGHTS RMS*/
	addKernelArg(16, 0, sizeof(cl_mem), (void*)&memobjNNParamsVector);
	addKernelArg(16, 1, sizeof(cl_mem), (void*)&memobjGradientVector);
	addKernelArg(16, 2, sizeof(cl_mem), (void*)&memobjLearningParameter);
}

void NNKernel::setImageIndex(int index)
{
	cl_int numSampleTemp = index;
	addKernelArg(6, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(5, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(0, 1, sizeof(cl_int), (void*)&numSampleTemp);
	addKernelArg(13, 1, sizeof(cl_int), (void*)&numSampleTemp);
}

void NNKernel::readNNParams()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjNNParamsVector, CL_TRUE, 0, NN_WEIGHT_SIZE * sizeof(float), (void*)nnParams, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readActivations()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjActivationVector, CL_TRUE, 0, NN_ACTIVATION_SIZE * sizeof(float), (void*)activations, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readGradient()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE * sizeof(float), (void*)gradient, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readActivationDelta()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjActivationDeltaVector, CL_TRUE, 0, NN_ACTIVATION_SIZE * sizeof(float), (void*)activationDeltas, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readCost()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjCost, CL_TRUE, 0, sizeof(float), (void*)&cost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readLearningParams()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, memobjLearningParameter, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), (void*)learningParameter, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void NNKernel::readBuffers()
{
	readNNParams();
	readActivations();
	readGradient();
	readActivationDelta();
	readCost();
	readLearningParams();
}

void NNKernel::calculateActivationsLayer1()
{
	totalWorkItems = NN_LAYER_1_SIZE;
	enqueueKernel(0);
	//readActivations();
}
void NNKernel::calculateActivationsLayer2()
{
	totalWorkItems = NN_LAYER_2_SIZE;
	enqueueKernel(1);
	//readActivations();
}
void NNKernel::calculateActivationsLayerOutput()
{
	totalWorkItems = NN_OUTPUT_SIZE;
	enqueueKernel(2);
	//readActivations();
}

void NNKernel::calculateActivationsDeltaLayer1()
{
	totalWorkItems = NN_LAYER_1_SIZE;
	enqueueKernel(3);
	//readActivationDelta();
}
void NNKernel::calculateActivationsDeltaLayer2()
{
	totalWorkItems = NN_LAYER_2_SIZE;
	enqueueKernel(4);
	//readActivationDelta();
}
void NNKernel::calculateActivationsDeltaLayerOutput()
{
	totalWorkItems = NN_OUTPUT_SIZE;
	enqueueKernel(5);
	//readActivationDelta();
}

void NNKernel::addGradientWeightLayer1()
{
	totalWorkItems = NN_LAYER_1_WEIGHT_SIZE;
	enqueueKernel(6);
	//readGradient();
}
void NNKernel::addGradientWeightLayer2()
{
	totalWorkItems = NN_LAYER_2_WEIGHT_SIZE;
	enqueueKernel(7);
	//readGradient();
}
void NNKernel::addGradientWeightLayerOutput()
{
	totalWorkItems = NN_LAYER_OUTPUT_WEIGHT_SIZE;
	enqueueKernel(8);
	//readGradient();
}

void NNKernel::addGradientBiasLayer1()
{
	totalWorkItems = NN_LAYER_1_BIAS_SIZE;
	enqueueKernel(9);
	//readGradient();
}
void NNKernel::addGradientBiasLayer2()
{
	totalWorkItems = NN_LAYER_2_BIAS_SIZE;
	enqueueKernel(10);
	//readGradient();
}
void NNKernel::addGradientBiasLayerOutput()
{
	totalWorkItems = NN_LAYER_OUTPUT_BIAS_SIZE;
	enqueueKernel(11);
	//readGradient();
}

void NNKernel::normalizeGradient()
{
	totalWorkItems = NN_WEIGHT_SIZE;
	enqueueKernel(14);
}

void NNKernel::computeCost()
{
	totalWorkItems = NN_OUTPUT_SIZE;
	enqueueKernel(13);
}

void NNKernel::updateLearningParams()
{
	totalWorkItems = NN_WEIGHT_SIZE;
	enqueueKernel(15);
}

void NNKernel::updateWeightsRMS()
{
	totalWorkItems = NN_WEIGHT_SIZE;
	enqueueKernel(16);
}

void NNKernel::updateNNParams()
{
	totalWorkItems = NN_WEIGHT_SIZE;
	enqueueKernel(12);
}

void NNKernel::clearCost()
{
	cost = 0;
	clEnqueueWriteBuffer(__context->commandQueue, memobjCost, CL_TRUE, 0, sizeof(float), &cost, 0, NULL, NULL);
}

void NNKernel::clearLearningParameter()
{
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		learningParameter[i] = 0.0;
	}
	clEnqueueWriteBuffer(__context->commandQueue, memobjLearningParameter, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), learningParameter, 0, NULL, NULL);
}

void NNKernel::clearGradient()
{
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		gradient[i] = 0.0;
	}
	clEnqueueWriteBuffer(__context->commandQueue, memobjGradientVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), gradient, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

float NNKernel::gradientInnerProduct()
{
	float ip = 0.0;
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		ip += gradient[i] * gradient[i];
	}
	return ip;
}

float NNKernel::learningParamInnerProduct()
{
	float ip = 0.0;
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		ip += learningParameter[i] * learningParameter[i];
	}
	return ip;
}

float getWeightLayer1(float* w, int inputIndex, int layer1Index)
{
	return w[inputIndex*NN_LAYER_1_SIZE + layer1Index];
}
float getBiasLayer1(float* w, int layer1Index)
{
	return w[NN_LAYER_1_WEIGHT_SIZE + layer1Index];
}
float getWeightLayer2(float* w, int layer1Index, int layer2Index)
{
	int i = layer1Index*NN_LAYER_2_SIZE + layer2Index;
	i += NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE;
	return w[i];
}
float getBiasLayer2(float* w, int layer2Index)
{
	int i = layer2Index;
	i += NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE;
	return w[i];
}
float getWeightLayerOutput(float* w, int layer2Index, int layerOutputIndex)
{
	int i = layer2Index*NN_OUTPUT_SIZE + layerOutputIndex;
	i += NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE;
	return w[i];
}
float getBiasLayerOutput(float* w, int layerOutputIndex)
{
	int i = layerOutputIndex;
	i += NN_LAYER_1_WEIGHT_SIZE + NN_LAYER_1_BIAS_SIZE + NN_LAYER_2_WEIGHT_SIZE + NN_LAYER_2_BIAS_SIZE + NN_LAYER_OUTPUT_WEIGHT_SIZE;
	return w[i];
}

void NNKernel::exportNNParams(char* filePath)
{
	//Neural network sizes are assumed to be the size they are defined.
	readBuffers();
	std::fstream file;
	file.open(filePath, std::fstream::out | std::fstream::binary);
	if (!file)
		return;

	//write the weights to a file.
	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		float w = nnParams[i];
		file.write((char*)&w, 4);
	}
	
	file.close();
}

void NNKernel::importNNParams(char* filePath)
{
	std::fstream file;
	file.open(filePath, std::fstream::in | std::fstream::binary);
	
	if (!file)
		return;

	for (int i = 0; i < NN_WEIGHT_SIZE; i++)
	{
		float w;
		file.read((char*)&w, 4);
		nnParams[i] = w;
	}
	file.close();
	cl_int ret = clEnqueueWriteBuffer(__context->commandQueue, memobjNNParamsVector, CL_TRUE, 0, NN_WEIGHT_SIZE*sizeof(float), nnParams, 0, NULL, NULL);
}

void NNKernel::exportReport(char* filePath)
{
	time_t t = time(0);

	std::string fp(filePath);
	std::ofstream myfile;
	myfile.open(fp, std::ofstream::out);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";


	myfile << "Training time: ";
	myfile << std::to_string((float)elapsedTime / 60.0);
	myfile << "  min";
	myfile << "\n";


	myfile << "input vector size: ";
	myfile << std::to_string(NN_INPUT_SIZE);
	myfile << "\n";

	myfile << "layer 1 activation size: ";
	myfile << std::to_string(NN_LAYER_1_SIZE);
	myfile << "\n";

	myfile << "layer 2 activation size: ";
	myfile << std::to_string(NN_LAYER_2_SIZE);
	myfile << "\n";

	myfile << "output vector size: ";
	myfile << std::to_string(NN_OUTPUT_SIZE);
	myfile << "\n";

	myfile << "stochastic minibatch size: ";
	myfile << std::to_string(STOCHASTIC_SAMPLING_SIZE);
	myfile << "\n";

	myfile << "minibatch cost history:\n";
	for (int i = 0; i < miniBatchCostHistory.size(); i++)
	{
		myfile << std::to_string(miniBatchCostHistory[i]);
		myfile << "\n";
	}
	
	float c = totalCost();
	myfile << "total training set cost :";
	myfile << std::to_string(c);

	myfile.close();
}

void NNKernel::exportReport(char* filePath, vector<Mat*> &testImages, vector<unsigned char> &testLabels)
{
	time_t t = time(0);

	std::string fp(filePath);
	std::ofstream myfile;
	myfile.open(fp, std::ofstream::out);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";


	myfile << "Training time: ";
	myfile << std::to_string((float)elapsedTime / 60.0);
	myfile << "  min";
	myfile << "\n";


	myfile << "input vector size: ";
	myfile << std::to_string(NN_INPUT_SIZE);
	myfile << "\n";

	myfile << "layer 1 activation size: ";
	myfile << std::to_string(NN_LAYER_1_SIZE);
	myfile << "\n";

	myfile << "layer 2 activation size: ";
	myfile << std::to_string(NN_LAYER_2_SIZE);
	myfile << "\n";

	myfile << "output vector size: ";
	myfile << std::to_string(NN_OUTPUT_SIZE);
	myfile << "\n";

	myfile << "stochastic minibatch size: ";
	myfile << std::to_string(STOCHASTIC_SAMPLING_SIZE);
	myfile << "\n";

	myfile << "minibatch cost history:\n";
	for (int i = 0; i < miniBatchCostHistory.size(); i++)
	{
		myfile << std::to_string(miniBatchCostHistory[i]);
		myfile << "\n";
	}

	float c = totalCost();
	myfile << "total training set cost :";
	myfile << std::to_string(c);
	myfile << "\n";

	float errorRate = test(testImages, testLabels);
	myfile << "error rate:";
	myfile << std::to_string(errorRate);
	myfile << "\n";

	myfile.close();
}