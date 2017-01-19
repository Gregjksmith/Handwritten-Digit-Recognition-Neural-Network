/* NEURAL NETWORK KERNEL 

 AUTHOR: Greg Smith, 2016

OpenCL Neural Network kernels.
Trains a neural network to minimize the L2 norm error between f(x) and the output vector.
x is the input vector, N samples where each sample is size 'INPUT_SIZE'.
f(x) is the neural network output.
	2 Layers, layers are of size 'LAYER_1_SIZE' and 'LAYER_2_SIZE' respectively.
output vector is N samples of 'OUTPUT_SIZE' size.

'LEARNING_RATE_EPS' specifies a constant that controls the apative learning rate. A large 'LEARNING_RATE_EPS' may converge
faster or may display violent oscillations. Careful parameter selection is required.
*/

#define INPUT_SIZE 28*28

#define LAYER_1_SIZE 400
#define LAYER_2_SIZE 200

#define OUTPUT_SIZE 10

#define LEARNING_RATE_EPS 0.1
#define LEARNING_DECAY_RATE 0.5
#define DELTA 0.000001

/* WEIGHT STRUCTURE */
typedef struct _NNweights
{
	float layer1Weights[INPUT_SIZE*LAYER_1_SIZE];
	float layer1Bias[LAYER_1_SIZE];

	float layer2Weights[LAYER_1_SIZE*LAYER_2_SIZE];
	float layer2Bias[LAYER_2_SIZE];

	float layerOutputWeights[LAYER_2_SIZE*OUTPUT_SIZE];
	float layerOutputBias[OUTPUT_SIZE];
}NNweights;

/* ACTIVATION STRUCTURE */
typedef struct _Activation
{
	float layer1Activation[LAYER_1_SIZE];
	float layer2Activation[LAYER_2_SIZE];
	float layerOutputActivation[OUTPUT_SIZE];
}Activation;

/* ACTIVATION DELTA STRUCTURE */
typedef struct _ActivationDelta
{
	float layer1ActivationDelta[LAYER_1_SIZE];
	float layer2ActivationDelta[LAYER_2_SIZE];
	float layerOutputActivationDelta[OUTPUT_SIZE];
}ActivationDelta;

/* GRADIENT STRUCTURE */
typedef struct _Gradient
{
	float layer1Weights[INPUT_SIZE*LAYER_1_SIZE];
	float layer1Bias[LAYER_1_SIZE];

	float layer2Weights[LAYER_1_SIZE*LAYER_2_SIZE];
	float layer2Bias[LAYER_2_SIZE];

	float layerOutputWeights[LAYER_2_SIZE*OUTPUT_SIZE];
	float layerOutputBias[OUTPUT_SIZE];
}Gradient;

/* ================================== HEADERS ==========================================*/

/*
float sampleImage
gets the input image pixel intensity from the input image vector.
@param float* input : input image vector.
@param int imageIndex : index of the image in the image vector.
@param int pixelIndex : index of the pixel within an input image.
*/
float sampleImage(__global float* input, int imageIndex, int pixelIndex);

/*
float sampleOutput
samples the output vector.
@param float* output : output vector.
@param int outputIndex : output vector index.
@param int layerIndex : activation index within the output layer.
*/
float sampleOutput(__global float* output, int outputIndex, int layerIndex);

/*
void getWeightIndexLayer1
gets the input index and layer1Index from the corresponding global weight index.
@param int weightIndex : global vector weight index.
@param int* inputIndex : input activation index.
@param int* layer1Index : layer 1 activation index.  
*/
void getWeightIndexLayer1(int weightIndex, int* inputIndex, int* layer1Index);

/*
void getWeightIndexLayer2
gets the layer1 index and the layer 2 index from the global weight index.
@param weightIndex : global vector weight index.
@param int* layer1Index : layer 1 activation index.
@param int* layer2Index : layer 2 activation index.
*/
void getWeightIndexLayer2(int weightIndex, int* layer1Index, int* layer2Index);

/*
void getWeightIndexOutput
gets the layer2 index and the output index from the global weight index.
@param weightIndex : global vector weight index.
@param int* layer2Index : layer 2 activation index.
@param int* outputIndex : output activation index.
*/
void getWeightIndexOutput(int weightIndex, int* layer2Index, int* outputIndex);

/*
float getWeightLayer1
gets the NN weight connecting the inputIndex and layer1Index.
@param NNweight* weight : NN weight vector
@param int inputIndex : input activation index.
@param int layer1Index : layer 1 activation index.
*/
float getWeightLayer1(__global NNweights* weights, int inputIndex, int layer1Index);

/*
void getBiasLayer1
gets the bias at a layer 1 activation.
@param NNweight* weight : NN weight vector
@param int* layer1Index : layer 1 activation index.
*/
float getBiasLayer1(__global NNweights* weights, int layer1Index);


/*
float getWeightLayer2
gets the NN weight connecting the layer1Index and layer2Index.
@param NNweight* weight : NN weight vector
@param int layer1Index : layer 1 activation index.
@param int layer2Index : layer 2 activation index.
*/
float getWeightLayer2(__global NNweights* weights, int layer1Index, int layer2Index);

/*
void getBiasLayer2
gets the bias at a layer 2 activation.
@param NNweight* weight : NN weight vector
@param int* layer2Index : layer 2 activation index.
*/
float getBiasLayer2(__global NNweights* weights, int layer2Index);

/*
float getWeightLayerOutput
gets the NN weight connecting the layer2Index and layerOutputIndex.
@param NNweight* weight : NN weight vector
@param int layer2Index : layer 2 activation index.
@param int layerOutputIndex : layer output activation index.
*/
float getWeightLayerOutput(__global NNweights* weights, int layer2Index, int layerOutputIndex);

/*
float getBiasLayerOutput
gets the NN bias at an output layer index.
@param NNweight* weight : NN weight vector
@param int layerOutputIndex : layer output activation index.
*/
float getBiasLayerOutput(__global NNweights* weights, int layerOutputIndex);

/*
float getGradientWeightLayer1
samples the weight gradient from the gradient vector in layer 1.
@param int inputIndex
@param int layer1Index
*/
float getGradientWeightLayer1(__global Gradient* gradient, int inputIndex, int layer1Index);

/*
float getGradientBiasLayer1
samples the bias gradient from the gradient vector in layer 1.
@param int layer1Index
*/
float getGradientBiasLayer1(__global Gradient* gradient, int layer1Index);

/*
float getGradientWeightLayer2
samples the weight gradient from the gradient vector in layer 2.
@param int layer1Index
*/
float getGradientWeightLayer2(__global Gradient* gradient, int layer1Index, int layer2Index);

/*
float getGradientBiasLayer1
samples the bias gradient from the gradient vector in layer 2.
@param int layer2Index
*/
float getGradientBiasLayer2(__global Gradient* gradient, int layer2Index);

/*
float getGradientWeightLayerOutput
samples the weight gradient from the gradient vector in layer output.
@param int layer2Index
@param int layerOutputIndex
*/
float getGradientWeightLayerOutput(__global Gradient* gradient, int layer2Index, int layerOutputIndex);

/*
float getGradientBiasLayerOutput
samples the bias gradient from the gradient vector in layer output.
@param int layerOutputIndex
*/
float getGradientBiasLayerOutput(__global Gradient* gradient, int layerOutputIndex);

/*
float activationFunction
sigmoid function.
@param float x
*/
float activationFunction(float x);

/*
float activation Derivative
derivative of 'activationFunction' evaluated at x
@param float x
*/
float activationDerivative(float x);

/*
float tanHyperbolicDerivative
derivative of tanh() evaluated at x
@param float x
*/
float tanHyperbolicDerivative(float x);



/*
kernel void activationLayer1
computes the activations at layer 1
@param float* input : input vector
@param int inputIndex
@param Activation* activation
@param NNweights* weights
*/
__kernel void activationLayer1(__global float* input, int inputIndex, __global Activation* activation, __global NNweights* weights);

/*
kernel void activationLayer2
computes the activations at layer 2
@param Activation* activation
@param NNweights* weights
*/
__kernel void activationLayer2(__global Activation* activation, __global NNweights* weights);

/*
kernel void activationLayerOutput
computes the activations at the output layer
@param Activation* activation
@param NNeights* weights
*/
__kernel void activationLayerOutput(__global Activation* activation, __global NNweights* weights);

/*
kernel void activationOutputDelta
computes the activations deltas (the derivative of the output w.r.t the activation) at the output layer.
@param float* output : output vector
@param int outputIndex
@param Activation* activation
@param ActivationDelta* activationDelta
*/
__kernel void activationOutputDelta(__global float* output, int outputIndex, __global Activation* activation, __global ActivationDelta* activationDelta);

/*
kernel void activationLayer2Delta
computes the activations deltas (the derivative of the output w.r.t the activation) at layer 2.
@param Activation* activation
@param ActivationDelta* activationDelta
@param NNweights* weights
*/
__kernel void activationLayer2Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights);

/*
kernel void activationLayer1Delta
computes the activations deltas (the derivative of the output w.r.t the activation) at layer 1.
@param Activation* activation
@param ActivationDelta* activationDelta
@param NNweights* weights
*/
__kernel void activationLayer1Delta( __global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights);



/*
kernel void addGradientWeightLayer1
computes the weight in layer 1gradient and adds it to the gradient vector.
@param float* input
@param int inputIndex
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayer1(__global float* input, int inputIndex, __global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientWeightLayer2
computes the weight gradient in layer 2and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientWeightLayerOutput
computes the weight gradient in the output layer and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientWeightLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayer1
computes the bias gradient in layer 1 and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayer1(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayer2
computes the bias gradient in layer 2 and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void addGradientBiasLayerOutput
computes the bias gradient in the output layer and adds it to the gradient vector.
@param Activation* activation
@param NNeights* weights
@param ActivationDelta* activationDelta
@param Gradient* gradient
*/
__kernel void addGradientBiasLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient);

/*
kernel void normalizeGradient
divides the gradient by the number of samples.
@param float* gradient
@param int numSamples
*/
__kernel void normalizeGradient(__global float* gradient, int numSamples);

/*
kernel void updateNNparams
updates the Neural netowrk weights by the learningRate times the negative gradient
@param float* weights
@param float* gradient
@param float learningRate
*/
__kernel void updateNNparams(__global float* weights, __global float* gradient, float learningRate);

/*
kernel void updateNNParamsRMS
updates the Neural netowrk weights using the RMSProp algorithm
@param float* weights
@param float* gradient
@param float* learningParams : vector of learning rates, one for each NNparam
*/
__kernel void updateNNParamsRMS(__global float* weights, __global float* gradient, __global float* learningParams);

/*
kernel void cost
Computes the L2 norm between the output activations and the outputVector at index outputIndex. Cost is divided by 2, such that
the gradient is not scales by 2. The cost is then added to the buffer returnCost
@param float* outputVector
@param int outputIndex
@param Activation* activation
@param float* returnCost : return cost is a floating point buffer of size 1, used to compute the average cost over N samples.
*/
__kernel void cost(__global float* outputVector, int outputIndex, __global Activation* activation, __global float* returnCost);

/*
kerenl void updateLearningParams
Computes the running average of gradient lengths. Used as part of the RMSProp algorithm.
@param float* learningParams.
@param float* gradient.
*/
__kernel void updateLearningParams(__global float* learningParams, __global float* gradient);

/* ==================================================================================== */

float sampleImage(__global float* input, int imageIndex, int pixelIndex)
{
	int i = imageIndex*INPUT_SIZE + pixelIndex;
	return input[i];
}

float sampleOutput(__global float* output, int outputIndex, int layerIndex)
{
	int i = outputIndex*OUTPUT_SIZE + layerIndex;
	return output[i];
}

float getWeightLayer1(__global NNweights* weights, int inputIndex, int layer1Index)
{
	int i = inputIndex*LAYER_1_SIZE + layer1Index;
	return weights->layer1Weights[i];
}

float getBiasLayer1(__global NNweights* weights, int layer1Index)
{
	return weights->layer1Bias[layer1Index];
}

float getWeightLayer2(__global NNweights* weights, int layer1Index, int layer2Index)
{
	int i = layer1Index*LAYER_2_SIZE + layer2Index;
	return weights->layer2Weights[i];
}

float getBiasLayer2(__global NNweights* weights, int layer2Index)
{
	return weights->layer2Bias[layer2Index];
}

float getWeightLayerOutput(__global NNweights* weights, int layer2Index, int layerOutputIndex)
{
	int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;
	return weights->layerOutputWeights[i];
}

float getBiasLayerOutput(__global NNweights* weights, int layerOutputIndex)
{
	return weights->layerOutputBias[layerOutputIndex];
}


float getGradientWeightLayer1(__global Gradient* gradient, int inputIndex, int layer1Index)
{
	int i = inputIndex*LAYER_1_SIZE + layer1Index;
	return gradient->layer1Weights[i];
}
float getGradientBiasLayer1(__global Gradient* gradient, int layer1Index)
{
	return gradient->layer1Bias[layer1Index];
}

float getGradientWeightLayer2(__global Gradient* gradient, int layer1Index, int layer2Index)
{
	int i = layer1Index*LAYER_2_SIZE + layer2Index;
	return gradient->layer2Weights[i];
}
float getGradientBiasLayer2(__global Gradient* gradient, int layer2Index)
{
	return gradient->layer2Bias[layer2Index];
}

float getGradientWeightLayerOutput(__global Gradient* gradient, int layer2Index, int layerOutputIndex)
{
	int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;
	return gradient->layerOutputWeights[i];
}
float getGradientBiasLayerOutput(__global Gradient* gradient, int layerOutputIndex)
{
	return gradient->layerOutputBias[layerOutputIndex];
}

float activationFunction(float x)
{
	float s = 1.0/(1.0 + exp(-x));
	return s;
}

float activationDerivative(float x)
{
	float s = activationFunction(x);
	s = s*(1.0 - s);
	return s;
}

float tanHyperbolicDerivative(float x)
{
	float tanhResult = tanh(x);
	return (1.0 - tanhResult*tanhResult);
}

__kernel void activationLayer1(__global float* input, int inputIndex, __global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	float activationSum = 0.0;
	for(int pixelIndex = 0; pixelIndex < INPUT_SIZE; pixelIndex++)
	{
		float im = sampleImage(input, inputIndex, pixelIndex);
		float w = getWeightLayer1(weights, pixelIndex, activationId);
		activationSum += im*w;
	}
	activationSum += getBiasLayer1(weights, activationId);
	activationSum = activationFunction(activationSum);
	activation->layer1Activation[activationId] = activationSum;
}

__kernel void activationLayer2(__global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	float activationSum = 0.0;
	for(int layer1Index = 0; layer1Index < LAYER_1_SIZE; layer1Index++)
	{
		float act1 = activation->layer1Activation[layer1Index];
		float w = getWeightLayer2(weights, layer1Index, activationId);
		activationSum += act1*w;
	}
	activationSum += getBiasLayer2(weights, activationId);
	activationSum = activationFunction(activationSum);
	activation->layer2Activation[activationId] = activationSum;
}

__kernel void activationLayerOutput(__global Activation* activation, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	float activationSum = 0.0;
	for(int layer2Index = 0; layer2Index < LAYER_2_SIZE; layer2Index++)
	{
		float act2 = activation->layer2Activation[layer2Index];
		float w = getWeightLayerOutput(weights, layer2Index, activationId);
		activationSum += act2*w;
	}
	activationSum += getBiasLayerOutput(weights, activationId);
	activationSum = activationFunction(activationSum);
	activation->layerOutputActivation[activationId] = activationSum;
}




__kernel void activationOutputDelta(__global float* output, int outputIndex, __global Activation* activation, __global ActivationDelta* activationDelta)
{
	int activationId = get_global_id(0);
	float delta = 0.0;
	float activationSample = activation->layerOutputActivation[activationId];
	delta = (activationSample - sampleOutput(output, outputIndex, activationId))*activationDerivative(activationSample);
	activationDelta->layerOutputActivationDelta[activationId] = delta;
}

__kernel void activationLayer2Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	float delta = 0.0;
	float activationSample;
	float weightSample;
	for(int outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++)
	{
		activationSample = activationDelta->layerOutputActivationDelta[outputIndex];
		weightSample = getWeightLayerOutput(weights, activationId, outputIndex);
		delta += activationSample*weightSample;
	}
	activationSample = activation->layer2Activation[activationId];
	delta = delta*activationDerivative(activationSample);
	activationDelta->layer2ActivationDelta[activationId] = delta;
}

__kernel void activationLayer1Delta(__global Activation* activation, __global ActivationDelta* activationDelta, __global NNweights* weights)
{
	int activationId = get_global_id(0);
	float delta = 0.0;
	float activationSample;
	float weightSample;
	for(int layer2Index = 0; layer2Index < LAYER_2_SIZE; layer2Index++)
	{
		activationSample = activationDelta->layer2ActivationDelta[layer2Index];
		weightSample = getWeightLayer2(weights, activationId, layer2Index);
		delta += activationSample*weightSample;
	}
	activationSample = activation->layer1Activation[activationId];
	delta = delta*activationDerivative(activationSample);
	activationDelta->layer1ActivationDelta[activationId] = delta;
}

void getWeightIndexLayer1(int weightIndex, int* inputIndex, int* layer1Index)
{
	//int i = inputIndex*LAYER_1_SIZE + layer1Index;

	*inputIndex = weightIndex / LAYER_1_SIZE;
	*layer1Index = weightIndex - ((*inputIndex)*LAYER_1_SIZE);
}

void getWeightIndexLayer2(int weightIndex, int* layer1Index, int* layer2Index)
{
	//int i = layer1Index*LAYER_2_SIZE + layer2Index;

	*layer1Index = weightIndex / LAYER_2_SIZE;
	*layer2Index = weightIndex - ((*layer1Index)*LAYER_2_SIZE);
}

void getWeightIndexOutput(int weightIndex, int* layer2Index, int* outputIndex)
{
	//int i = layer2Index*OUTPUT_SIZE + layerOutputIndex;

	*layer2Index = weightIndex / OUTPUT_SIZE;
	*outputIndex = weightIndex - ((*layer2Index)*OUTPUT_SIZE);
}



__kernel void addGradientWeightLayer1(__global float* input, int inputIndex, __global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	
	int layer1ActivationIndex;
	int inputActivation;
	getWeightIndexLayer1(gradientId, &inputActivation, &layer1ActivationIndex);

	float grad = gradient->layer1Weights[gradientId];
	float alInput = sampleImage(input, inputIndex, inputActivation);
	float del1 = activationDelta->layer1ActivationDelta[layer1ActivationIndex];

	grad = grad + del1*alInput;

	// gradient = gradient + input*deltaLayer1
	gradient->layer1Weights[gradientId] = grad;
}

__kernel void addGradientWeightLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	
	int layer2ActivationIndex;
	int layer1ActivationIndex;
	getWeightIndexLayer2(gradientId, &layer1ActivationIndex, &layer2ActivationIndex);

	float grad = gradient->layer2Weights[gradientId];
	float al1 = activation->layer1Activation[layer1ActivationIndex];
	float del2 = activationDelta->layer2ActivationDelta[layer2ActivationIndex];

	grad = grad + del2*al1;

	// gradient = gradient + activationLayer1*deltaLayer2
	gradient->layer2Weights[gradientId] = grad;
}

__kernel void addGradientWeightLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	
	int outputActivationIndex;
	int layer2ActivationIndex;
	getWeightIndexOutput(gradientId, &layer2ActivationIndex, &outputActivationIndex);
	
	float grad = gradient->layerOutputWeights[gradientId];
	float al2 = activation->layer2Activation[layer2ActivationIndex];
	float delOut = activationDelta->layerOutputActivationDelta[outputActivationIndex];

	grad = grad + delOut*al2;

	// gradient = gradient + activationLayer2*deltaOutput
	gradient->layerOutputWeights[gradientId] = grad;

}

__kernel void addGradientBiasLayer1(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	gradient->layer1Bias[gradientId] += activationDelta->layer1ActivationDelta[gradientId];
}

__kernel void addGradientBiasLayer2(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	gradient->layer2Bias[gradientId] += activationDelta->layer2ActivationDelta[gradientId];
}

__kernel void addGradientBiasLayerOutput(__global Activation* activation, __global NNweights* weights, __global ActivationDelta* activationDelta, __global Gradient* gradient)
{
	int gradientId = get_global_id(0);
	gradient->layerOutputBias[gradientId] += activationDelta->layerOutputActivationDelta[gradientId];
}

__kernel void updateNNparams(__global float* weights, __global float* gradient, float learningRate)
{
	int gradientId = get_global_id(0);
	float w = weights[gradientId];
	float g = gradient[gradientId];
	w = w - g*learningRate;
	weights[gradientId] = w;
}

__kernel void updateNNParamsRMS(__global float* weights, __global float* gradient, __global float* learningParams)
{
	int gradientId = get_global_id(0);
	float r = learningParams[gradientId];
	float g = gradient[gradientId];
	float learnRate = (LEARNING_RATE_EPS)/sqrt(DELTA + r);
	learnRate = min(learnRate,200.0);
	float gradDelta = -g*learnRate;

	weights[gradientId] = weights[gradientId] + gradDelta;
}

__kernel void cost(__global float* outputVector, int outputIndex, __global Activation* activation, __global float* returnCost)
{
	float c = 0.0;
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		float err = activation->layerOutputActivation[i] - outputVector[outputIndex*OUTPUT_SIZE + i];
		c = c + err*err;
	}
	returnCost[0] += c/2.0;
}

__kernel void normalizeGradient(__global float* gradient, int numSamples)
{
	int gradientId = get_global_id(0);
	gradient[gradientId] = gradient[gradientId] / (float)numSamples;
}

__kernel void updateLearningParams(__global float* learningParams, __global float* gradient)
{
	int index = get_global_id(0);
	float r = learningParams[index];
	float g = gradient[index];

	learningParams[index] = LEARNING_DECAY_RATE*r + (1.0 - LEARNING_DECAY_RATE)*(g*g); 
}