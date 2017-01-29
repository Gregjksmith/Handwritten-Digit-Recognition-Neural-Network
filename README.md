# Handwritten Digit Recognition using Neural Networks

## Training Set

Handwritten Digits taken from [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

![Samples from the MNIST Database](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/mnistSet.png?raw=true)

## Architecture

![Neural Network with 2 Hidden Layers](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/TwoLayerNeuralNetwork.png?raw=true)

1. Input : handwritten vectorized image, 784 nodes.
2. Layer 1 : fully connected, 50 nodes. 39200 total weights. 50 biases. Sigmoid activation function.
3. Layer 2 : fully connected, 50 nodes. 2500 total weights. 50 biases. Sigmoid activation function.
4. Output layer : fully connected, 10 nodes. 500 total weights. 10 biases. Sigmoid activation function.

total: 42310 total tunable parameters.

## Learning Method

mini-batch gradient descent backpropagation.

### Cost Function

Given a set of training vectors {xn} where n = 1...N, with corresponding target vectors {tn}, we seek to minimize the cost function E(w).

![Cost](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/CostFunction.gif?raw=true)

w is the set of neural network parameters and y(.) models the output of the neural network.

Each neuron (or activation) in the network is a weighted sum of the activations in the previous layer with an added bias. We also apply a non-linear
activation function to each node h(t), capping the activations. Here we use the sigmoid function.

![sigmoid](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/sigmoid.gif?raw=true)

The activations are calculated by:

![Activations](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/Activation.gif?raw=true)

where k is the neuron index of a given layer and j is the neuron index in the layer preceding k.

### Algorithm

Hyperparameters: learning rate (epsilon), number of iterations/epochs (Iterations), mini-batch sampling size (N).

1. for i = 1 ... Iterations
2.  stochastically sample a minibatch of training images (N)
3.  compute the gradient of the cost with respect to the neural network parameters w. Use [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
5.  update the neural network parameters. ![](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/update.gif?raw=true)
7.  end

## Results

### Hyperparameters

| Hyperparameter        	| Value         |
| --------------------------|:-------------:|
| Input Layer Neurons   	| 784		    |
| Layer 1 Neurons			| 50      		|
| Layer 2 Neurons			| 50     		|
| Output Layer Neurons  	| 10     		|
| Mini-batch size			| 200			|
| epochs                	| 5000			|


### Training Time and Error Rate

| --------------------------|:-------------:|
| Training Time				| 100 minutes	|
| Total Training Set Cost   | 0.021939		|
| Error Rate   				| 3.49%			|

![cost results]https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/NeuralNetworkTrainingGraph.png?raw=true