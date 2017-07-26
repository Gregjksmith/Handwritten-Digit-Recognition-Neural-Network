/* 

-----------------------------
	C1 feature map:

	5*5*6 feature map weights
	6 biases

	float weights[150];
	vectorized order : row, column, feature map 

	float biases[6];
-----------------------------

-----------------------------
	S2 feature map:

	6 weights;
	6 biases;

	float weights[6];
	float biases[6];
-----------------------------

-----------------------------
	C3 convolution layer:

	10*6*25 weights
	16 biases.

	float weights[1500];
	vectorized order: kernel index convolved with S2 (i.e. c3 index), s2 index, kernel row, kernel column, 

	float biases[16];
-----------------------------

-----------------------------
	S4 subsampling layer

	16 weights;
	16 biases;

	float weights[16];
	float biases[16];
-----------------------------

-----------------------------
	C5 convolution layer

	fully connect each pixel in each feature map to 120 different nodes.

	16*5*5*120 weights;
	120 biases;

	float weights[48000]
	vectorized order: C5 index, s4 feature map index, S4 row index, S4 column index,

	float biases[120]
-----------------------------

-----------------------------
	F6 layer

	fully connected

	84*120 weights;
	184 biases;

	float weight[10080]
	vectorized order: F6 index, C5 index
	float biases[184]
-----------------------------

-----------------------------
	output

	rbf classifier

	10*84 'rbf centers';

	float centers[840];
	vectorized order: output index, F6 index.
-----------------------------

units in layers up to F6, the activation function applied.
xi = f(ai);
f(a) = Atanh(Sa); A = 1.7159;

*/

#define CONST_A 1.7159

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5

#define C_1_WEIGHT_SIZE 150
#define C_1_BIAS_SIZE 6

#define S_2_WEIGHT_SIZE 6
#define S_2_BIAS_SIZE 6

#define C_3_KERNELS_PER_S2 10
#define C_3_WEIGHT_SIZE 1500
#define C_3_BIAS_SIZE 16


#define S_4_WEIGHT_SIZE 16
#define S_4_BIAS_SIZE 16

#define C_5_WEIGHT_SIZE 48000
#define C_5_BIAS_SIZE 120

#define F_6_WEIGHT_SIZE 10080
#define F_6_BIAS_SIZE 184

#define OUTPUT_RBF_CENTERS 840


typedef struct __CNNparams
{
	float c1Weight[C_1_WEIGHT_SIZE];
	float c1Bias[C_1_BIAS_SIZE];

	float s2Weight[S_2_WEIGHT_SIZE];
	float s2Bias[S_2_BIAS_SIZE];

	float c3Weight[C_3_WEIGHT_SIZE];
	float c3Bias[C_3_BIAS_SIZE];

	float s4Weight[S_4_WEIGHT_SIZE];
	float s4Bias[S_4_BIAS_SIZE];

	float c5Weight[C_5_WEIGHT_SIZE];
	float c5Bias[C_5_BIAS_SIZE];

	float f6Weight[F_6_WEIGHT_SIZE];
	float f6Bias[F_6_BIAS_SIZE];

	float outputRBFCenter[OUTPUT_RBF_CENTERS];
} CNNparams;


#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32

#define FEATURE_MAP_C_1_WIDTH 28
#define FEATURE_MAP_C_1_HEIGHT 28
#define NUM_FEATURE_MAPS_C_1 6

#define FEATURE_MAP_S_2_WIDTH 14
#define FEATURE_MAP_S_2_HEIGHT 14
#define NUM_FEATURE_MAPS_S_2 6

#define FEATURE_MAP_C_3_WIDTH 10
#define FEATURE_MAP_C_3_HEIGHT 10
#define NUM_FEATURE_MAPS_C_3 16

#define FEATURE_MAP_S_4_WIDTH 5
#define FEATURE_MAP_S_4_HEIGHT 5
#define NUM_FEATURE_MAPS_S_4 16

#define NUM_FEATURE_MAPS_C_5 120

#define NUM_FEATURE_MAPS_F_6 84

#define NUM_FEATURE_MAPS_OUTPUT 10

#define IMAGE_SIZE_INPUT IMAGE_WIDTH*IMAGE_HEIGHT
#define ACTIVATION_SIZE_C_1 NUM_FEATURE_MAPS_C_1*FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT
#define ACTIVATION_SIZE_S_2 NUM_FEATURE_MAPS_S_2*FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT
#define ACTIVATION_SIZE_C_3 NUM_FEATURE_MAPS_C_3*FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT
#define ACTIVATION_SIZE_S_4 NUM_FEATURE_MAPS_S_4*FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT
#define ACTIVATION_SIZE_C_5 NUM_FEATURE_MAPS_C_5
#define ACTIVATION_SIZE_F_6 NUM_FEATURE_MAPS_F_6
#define ACTIVATION_SIZE_OUTPUT NUM_FEATURE_MAPS_OUTPUT

typedef struct __Activations
{
	float activationC1[ACTIVATION_SIZE_C_1];
	float activationS2[ACTIVATION_SIZE_S_2];
	float activationC3[ACTIVATION_SIZE_C_3];
	float activationS4[ACTIVATION_SIZE_S_4];
	float activationC5[ACTIVATION_SIZE_C_5];
	float activationF6[ACTIVATION_SIZE_F_6];
	float activationOutput[ACTIVATION_SIZE_OUTPUT];
}Activations;


// HEADERS
__kernel void activationC1(__global float* input, int sampleIndex, __global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationS2(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationC3(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationS4(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationC5(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationF6(__global Activations* activations, __global CNNparams* cnnParams);
__kernel void activationOutput(__global Activations* activations, __global CNNparams* cnnParams);

/* 
=======================================================================================
========================  GET CNN PARAM FUNCTIONS =====================================
=======================================================================================
*/

int getVectorizedImageIndex(int xIndex, int yIndex, int width, int height);
float getInputImageSample(__global float* image, int imageIndex, int xIndex, int yIndex);

float getC1Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapIndex);
float getC1Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getS2Weight(__global CNNparams* cnnParams, int featureMapIndex);
float getS2Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getC3Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC3Index, int featureMapS2Index);
float getC3Bias(__global CNNparams* cnnParams, int featureMapC3Index);
int getS2C3FeatureMapConnection(int featureMapC3Index, int kernelIndex);

float getS4Weight(__global CNNparams* cnnParams, int featureMapIndex);
float getS4Bias(__global CNNparams* cnnParams, int featureMapIndex);

float getC5Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC5Index, int featureMapS4Index);
float getC5Bias(__global CNNparams* cnnParams, int featureMapC5Index);

float getF6Weight(__global CNNparams* cnnParams, int featureMapF6Index, int featureMapC5Index);
float getF6Bias(__global CNNparams* cnnParams, int featureMapF6Index);

float getOutputRBFCenterWeight(__global CNNparams* cnnParam, int featureMapOutputIndex, int featureMapF6Index);

//-----------------------------

int getVectorizedImageIndex(int xIndex, int yIndex, int width, int height)
{
	int imageIndexLocal = yIndex*IMAGE_WIDTH + xIndex;
	return imageIndexLocal;
}

float getInputImageSample(__global float* image, int imageIndex, int xIndex, int yIndex)
{
	int imageIndexGlobal = imageIndex*IMAGE_WIDTH*IMAGE_HEIGHT;
	int imageIndexLocal = getVectorizedImageIndex(xIndex,yIndex,IMAGE_WIDTH,IMAGE_HEIGHT);
	return image[imageIndexGlobal + imageIndexLocal];
}

float getC1Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapIndex)
{
	int vectorizedIndex = getVectorizedImageIndex(xIndex,yIndex,KERNEL_WIDTH,KERNEL_HEIGHT);
	vectorizedIndex += featureMapIndex*KERNEL_WIDTH*KERNEL_HEIGHT;
	return cnnParams->c1Weight[vectorizedIndex];
}

float getC1Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->c1Bias[featureMapIndex];
}


float getS2Weight(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s2Weight[featureMapIndex];
}

float getS2Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s2Bias[featureMapIndex];
}

float getC3Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC3Index, int featureMapS2Index)
{
	//vectorized order: kernel row, kernel column, kernel index convolved with S2 (i.e. c3 index), s2 index.
	int vectorizedIndex = getVectorizedImageIndex(xIndex, yIndex, KERNEL_WIDTH, KERNEL_HEIGHT);
	int kernelSize = KERNEL_WIDTH*KERNEL_HEIGHT;
	int kernelIndex = featureMapC3Index + C_3_KERNELS_PER_S2*featureMapS2Index;
	vectorizedIndex += kernelSize*kernelIndex;
	return cnnParams->c3Weight[vectorizedIndex];
}

float getC3Bias(__global CNNparams* cnnParams, int featureMapC3Index)
{
	return cnnParams->c3Bias[featureMapC3Index];
}

int getS2C3FeatureMapConnection(int featureMapC3Index, int kernelIndex)
{
	switch(featureMapC3Index)
	{
		case 0:
		{
			if(kernelIndex == 0)
			{
				return 0;
			}
			else if(kernelIndex == 1)
			{
				return 1;
			}
			else if(kernelIndex == 2)
			{
				return 2;
			}
			else 
			{
				return -1;
			}

		}
		case 1:
		{
			if(kernelIndex == 0)
			{
				return 1;
			}
			else if(kernelIndex == 1)
			{
				return 2;
			}
			else if(kernelIndex == 2)
			{
				return 3;
			}
			else 
			{
				return -1;
			}

		}
		case 2:
		{
			if(kernelIndex == 0)
			{
				return 2;
			}
			else if(kernelIndex == 1)
			{
				return 3;
			}
			else if(kernelIndex == 2)
			{
				return 4;
			}
			else 
			{
				return -1;
			}

		}
		case 3:
		{
			if(kernelIndex == 0)
			{
				return 3;
			}
			else if(kernelIndex == 1)
			{
				return 4;
			}
			else if(kernelIndex == 2)
			{
				return 5;
			}
			else 
			{
				return -1;
			}

		}
		case 4:
		{
			if(kernelIndex == 0)
			{
				return 4;
			}
			else if(kernelIndex == 1)
			{
				return 5;
			}
			else if(kernelIndex == 2)
			{
				return 0;
			}
			else 
			{
				return -1;
			}

		}
		case 5:
		{
			if(kernelIndex == 0)
			{
				return 5;
			}
			else if(kernelIndex == 1)
			{
				return 0;
			}
			else if(kernelIndex == 2)
			{
				return 1;
			}
			else 
			{
				return -1;
			}

		}
		case 6:
		{
			if(kernelIndex == 0)
			{
				return 0;
			}
			else if(kernelIndex == 1)
			{
				return 1;
			}
			else if(kernelIndex == 2)
			{
				return 2;
			}
			else if(kernelIndex == 3)
			{
				return 3;
			}
			else 
			{
				return -1;
			}

		}
		case 7:
		{
			if(kernelIndex == 0)
			{
				return 1;
			}
			else if(kernelIndex == 1)
			{
				return 2;
			}
			else if(kernelIndex == 2)
			{
				return 3;
			}
			else if(kernelIndex == 3)
			{
				return 4;
			}
			else 
			{
				return -1;
			}

		}
		case 8:
		{
			if(kernelIndex == 0)
			{
				return 2;
			}
			else if(kernelIndex == 1)
			{
				return 3;
			}
			else if(kernelIndex == 2)
			{
				return 4;
			}
			else if(kernelIndex == 3)
			{
				return 5;
			}
			else 
			{
				return -1;
			}

		}
		case 9:
		{
			if(kernelIndex == 0)
			{
				return 3;
			}
			else if(kernelIndex == 1)
			{
				return 4;
			}
			else if(kernelIndex == 2)
			{
				return 5;
			}
			else if(kernelIndex == 3)
			{
				return 0;
			}
			else 
			{
				return -1;
			}

		}
		case 10:
		{
			if(kernelIndex == 0)
			{
				return 4;
			}
			else if(kernelIndex == 1)
			{
				return 5;
			}
			else if(kernelIndex == 2)
			{
				return 0;
			}
			else if(kernelIndex == 3)
			{
				return 1;
			}
			else 
			{
				return -1;
			}

		}
		case 11:
		{
			if(kernelIndex == 0)
			{
				return 5;
			}
			else if(kernelIndex == 1)
			{
				return 0;
			}
			else if(kernelIndex == 2)
			{
				return 1;
			}
			else if(kernelIndex == 3)
			{
				return 2;
			}
			else 
			{
				return -1;
			}

		}
		case 12:
		{
			if(kernelIndex == 0)
			{
				return 3;
			}
			else if(kernelIndex == 1)
			{
				return 4;
			}
			else if(kernelIndex == 2)
			{
				return 0;
			}
			else if(kernelIndex == 3)
			{
				return 1;
			}
			else 
			{
				return -1;
			}

		}
		case 13:
		{
			if(kernelIndex == 0)
			{
				return 4;
			}
			else if(kernelIndex == 1)
			{
				return 5;
			}
			else if(kernelIndex == 2)
			{
				return 1;
			}
			else if(kernelIndex == 3)
			{
				return 2;
			}
			else 
			{
				return -1;
			}

		}
		case 14:
		{
			if(kernelIndex == 0)
			{
				return 0;
			}
			else if(kernelIndex == 1)
			{
				return 2;
			}
			else if(kernelIndex == 2)
			{
				return 3;
			}
			else if(kernelIndex == 3)
			{
				return 5;
			}
			else 
			{
				return -1;
			}

		}
		case 15:
		{
			if(kernelIndex == 0)
			{
				return 0;
			}
			else if(kernelIndex == 1)
			{
				return 1;
			}
			else if(kernelIndex == 2)
			{
				return 2;
			}
			else if(kernelIndex == 3)
			{
				return 3;
			}
			else if(kernelIndex == 4)
			{
				return 4;
			}
			else if(kernelIndex == 5)
			{
				return 5;
			}
			else 
			{
				return -1;
			}

		}
	}
	return -1;
}

float getS4Weight(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s4Weight[featureMapIndex];
}

float getS4Bias(__global CNNparams* cnnParams, int featureMapIndex)
{
	return cnnParams->s4Bias[featureMapIndex];
}

float getC5Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC5Index, int featureMapS4Index)
{
	//120 * 16 * 5 * 5
	int kernelPixelIndex = getVectorizedImageIndex(xIndex, yIndex, FEATURE_MAP_S_4_WIDTH, FEATURE_MAP_S_4_HEIGHT);
	int vectorizedIndex = featureMapC5Index*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT*NUM_FEATURE_MAPS_S_4);
	vectorizedIndex += featureMapS4Index*(KERNEL_HEIGHT*KERNEL_WIDTH);
	vectorizedIndex += kernelPixelIndex;
	return cnnParams->c5Weight[vectorizedIndex];
}
float getC5Bias(__global CNNparams* cnnParams, int featureMapC5Index)
{
	return cnnParams->c5Bias[featureMapC5Index];
}

float getF6Weight(__global CNNparams* cnnParams, int featureMapF6Index, int featureMapC5Index)
{
	int vectorizedIndex = featureMapF6Index*NUM_FEATURE_MAPS_C_5 + featureMapC5Index;
	return cnnParams->f6Weight[vectorizedIndex];
}
float getF6Bias(__global CNNparams* cnnParams, int featureMapF6Index)
{
	return cnnParams->f6Bias[featureMapF6Index];
}

float getOutputRBFCenterWeight(__global CNNparams* cnnParam, int featureMapOutputIndex, int featureMapF6Index)
{
	int vectorizedIndex = featureMapOutputIndex*NUM_FEATURE_MAPS_F_6 + featureMapF6Index;
	return cnnParam->outputRBFCenter[vectorizedIndex];
}


/*
======================================================================================================
======================== GET ACTIVATION FUNCTIONS ====================================================
======================================================================================================
*/

float getActivationC1(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationS2(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationC3(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationS4(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex);
float getActivationC5(__global Activations* activation, int featureMapIndex);
float getActivationF6(__global Activations* activation, int featureMapIndex);
float getActivationOutput(__global Activations* activation, int featureMapIndex);

void getActivationIndiciesC1(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesS2(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesC3(int activationIndex, int* xIndex, int* yIndex, int* f);
void getActivationIndiciesS4(int activationIndex, int* xIndex, int* yIndex, int* f);

float getActivationC1(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int vImageIndex = getVectorizedImageIndex(xIndex, yIndex, FEATURE_MAP_C_1_WIDTH, FEATURE_MAP_C_1_HEIGHT);
	int vectorizedIndex = FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT*featureMapIndex + vImageIndex;
	return activation->activationC1[vectorizedIndex];
}

float getActivationS2(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int vImageIndex = getVectorizedImageIndex(xIndex, yIndex, FEATURE_MAP_S_2_WIDTH, FEATURE_MAP_S_2_HEIGHT);
	int vectorizedIndex = FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT*featureMapIndex + vImageIndex;
	return activation->activationS2[vectorizedIndex];
}

float getActivationC3(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int vImageIndex = getVectorizedImageIndex(xIndex, yIndex, FEATURE_MAP_C_3_WIDTH, FEATURE_MAP_C_3_HEIGHT);
	int vectorizedIndex = FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT*featureMapIndex + vImageIndex;
	return activation->activationC3[vectorizedIndex];
}

float getActivationS4(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)
{
	int vImageIndex = getVectorizedImageIndex(xIndex, yIndex, FEATURE_MAP_S_4_WIDTH, FEATURE_MAP_S_4_HEIGHT);
	int vectorizedIndex = FEATURE_MAP_S_4_HEIGHT*FEATURE_MAP_S_4_WIDTH*featureMapIndex + vImageIndex;
	return activation->activationS4[vectorizedIndex];
}

float getActivationC5(__global Activations* activation, int featureMapIndex)
{
	return activation->activationC5[featureMapIndex];
}

float getActivationF6(__global Activations* activation, int featureMapIndex)
{
	return activation->activationF6[featureMapIndex];
}

float getActivationOutput(__global Activations* activation, int featureMapIndex)
{
	return activation->activationOutput[featureMapIndex];
}


void getActivationIndiciesC1(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_C_1_WIDTH*FEATURE_MAP_C_1_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_C_1_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_C_1_WIDTH;
	*xIndex = tempIndex;
}

void getActivationIndiciesS2(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_S_2_WIDTH*FEATURE_MAP_S_2_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_S_2_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_S_2_WIDTH;
	*xIndex = tempIndex;
}
void getActivationIndiciesC3(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_C_3_WIDTH*FEATURE_MAP_C_3_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_C_3_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_C_3_WIDTH;
	*xIndex = tempIndex;
}
void getActivationIndiciesS4(int activationIndex, int* xIndex, int* yIndex, int* f)
{
	int tempIndex = activationIndex;
	*f = (tempIndex / (FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT));
	
	tempIndex = tempIndex - (*f)*(FEATURE_MAP_S_4_WIDTH*FEATURE_MAP_S_4_HEIGHT);
	*yIndex = (tempIndex / FEATURE_MAP_S_4_WIDTH);
	tempIndex = tempIndex - (*yIndex)*FEATURE_MAP_S_4_WIDTH;
	*xIndex = tempIndex;
}


__kernel void activationC1(__global float* input, int sampleIndex, __global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);

	int imageSampleCenterX;
	int imageSampleCenterY;
	int featureMapIndex;

	getActivationIndiciesC1(activationId, &imageSampleCenterX, &imageSampleCenterY, &featureMapIndex);
	int tX = imageSampleCenterX;
	int tY = imageSampleCenterY;
	int kernelHalfX = (KERNEL_WIDTH / 2);
	int kernelHalfY = (KERNEL_HEIGHT / 2);

	imageSampleCenterX = imageSampleCenterX + kernelHalfX;
	imageSampleCenterY = imageSampleCenterY + kernelHalfY;

	float activationSum = 0.0;

	//for(int imageIndex = 0; imageIndex < N; imageIndex++)
	{
		for(int x = 0; x <= KERNEL_WIDTH; x++ )
		{
			for(int y = 0; y <= KERNEL_HEIGHT; y++)
			{
				int kernelX = x;
				int kernelY = y;

				int imageX = x + imageSampleCenterX - kernelHalfX;
				int imageY = y + imageSampleCenterY - kernelHalfY;

				float ker = getC1Weight(cnnParams, x, y, featureMapIndex);
				float imSample = getInputImageSample(input, sampleIndex, imageX, imageY);
				activationSum = activationSum + ker*imSample;
			}
		}
	}

	activationSum = activationSum + getC1Bias(cnnParams,featureMapIndex);
	activationSum = CONST_A*tanh(activationSum);

	activations->activationC1[activationId] = activationSum;
}

__kernel void activationS2(__global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);
	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesS2(activationId, &xIndex, &yIndex, &featureMapIndex);

	int a1x1, a1x2, a1y1, a1y2;
	
	a1x1 = xIndex*2;
	a1x2 = a1x1 + 1;
	a1y1 = yIndex*2;
	a1y2 = a1y1 + 1;

	float activationSum = 0.0;
	activationSum  += getActivationC1(activations, a1x1, a1y1, featureMapIndex);
	activationSum  += getActivationC1(activations, a1x1, a1y2, featureMapIndex);
	activationSum  += getActivationC1(activations, a1x2, a1y1, featureMapIndex);
	activationSum  += getActivationC1(activations, a1x2, a1y2, featureMapIndex);
	activationSum = activationSum / 4.0; 

	activationSum = activationSum*getS2Weight(cnnParams, featureMapIndex);
	activationSum = activationSum + getS2Bias(cnnParams, featureMapIndex);
	activationSum = CONST_A*tanh(activationSum);

	activations->activationS2[activationId] = activationSum;
}

__kernel void activationC3(__global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);

	int imageSampleCenterX;
	int imageSampleCenterY;
	int featureMapC3;

	getActivationIndiciesC3(activationId, &imageSampleCenterX, &imageSampleCenterY, &featureMapC3);

	int kernelHalfX = (KERNEL_WIDTH / 2);
	int kernelHalfY = (KERNEL_HEIGHT / 2);

	imageSampleCenterX = imageSampleCenterX + kernelHalfX;
	imageSampleCenterY = imageSampleCenterY + kernelHalfY;

	float activationSum = 0.0;

	for(int kernelIndex = 0; kernelIndex < 6; kernelIndex++)
	{
		int featureMapS2 = getS2C3FeatureMapConnection(featureMapC3, kernelIndex);
		if(featureMapS2 >= 0)
		{
			for(int x = 0; x <= KERNEL_WIDTH; x++ )
			{
				for(int y = 0; y <= KERNEL_HEIGHT; y++)
				{

					int kernelX = x;
					int kernelY = y;

					int imageX = x + imageSampleCenterX - kernelHalfX;
					int imageY = y + imageSampleCenterY - kernelHalfY;

					//float getC3Weight(__global CNNparams* cnnParams, int xIndex, int yIndex, int featureMapC3Index, int featureMapS2Index);
					//getActivationS2(__global Activations* activation, int xIndex, int yIndex, int featureMapIndex)

					float ker = getC3Weight(cnnParams, kernelX, kernelY, featureMapC3, featureMapS2);
					float imSample = getActivationS2(activations, imageX, imageY, featureMapS2);
					activationSum = activationSum + ker*imSample;
					
				}
			}
		}
	}	

	activationSum = activationSum + getC3Bias(cnnParams,featureMapC3);
	activationSum = CONST_A*tanh(activationSum);

	activations->activationC3[activationId] = activationSum;
}

__kernel void activationS4(__global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);
	int xIndex;
	int yIndex;
	int featureMapIndex;
	getActivationIndiciesS4(activationId, &xIndex, &yIndex, &featureMapIndex);

	int a1x1, a1x2, a1y1, a1y2;
	
	a1x1 = xIndex*2;
	a1x2 = a1x1 + 1;
	a1y1 = yIndex*2;
	a1y2 = a1y1 + 1;

	float activationSum = 0.0;
	activationSum  += getActivationC3(activations, a1x1, a1y1, featureMapIndex);
	activationSum  += getActivationC3(activations, a1x1, a1y2, featureMapIndex);
	activationSum  += getActivationC3(activations, a1x2, a1y1, featureMapIndex);
	activationSum  += getActivationC3(activations, a1x2, a1y2, featureMapIndex);
	activationSum = activationSum / 4.0; 

	activationSum = activationSum*getS4Weight(cnnParams, featureMapIndex);
	activationSum = activationSum + getS4Bias(cnnParams, featureMapIndex);
	activationSum = CONST_A*tanh(activationSum);
	activations->activationS4[activationId] = activationSum;
}

__kernel void activationC5(__global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);

	float activationSum = 0.0;
	for(int i = 0; i < ACTIVATION_SIZE_S_4; i++)
	{
		activationSum += activations->activationS4[i]*cnnParams->c5Weight[activationId*ACTIVATION_SIZE_S_4 + i];
	}
	
	activationSum = activationSum + getC5Bias(cnnParams,activationId);
	activationSum = CONST_A*tanh(activationSum);
	activations->activationC5[activationId] = activationSum;
}

__kernel void activationF6(__global Activations* activations, __global CNNparams* cnnParams)
{
	size_t activationId = get_global_id(0);

	float activationSum = 0.0;
	for(int i=0; i < ACTIVATION_SIZE_C_5; i++)
	{
		activationSum += activations->activationC5[i]*cnnParams->f6Weight[activationId*ACTIVATION_SIZE_C_5 + i];
	}

	activationSum += getF6Bias(cnnParams,activationId);
	activations->activationF6[activationId] = activationSum;
}

__kernel void activationOutput(__global Activations* activations, __global CNNparams* cnnParams)
{
	//10 centers with of 84 dimensions.
	size_t activationId = get_global_id(0);
	float activationSum = 0.0;
	for(int i=0; i < NUM_FEATURE_MAPS_F_6; i++)
	{
		float actf6 = activations->activationF6[i];
		float rbfCenter = cnnParams->outputRBFCenter[activationId*NUM_FEATURE_MAPS_OUTPUT + i];
		float dist = actf6 - rbfCenter;
		dist = dist*dist;
		activationSum += dist;
	}
	activationSum = exp(-activationSum);
	activations->activationOutput[activationId] = activationSum;
}