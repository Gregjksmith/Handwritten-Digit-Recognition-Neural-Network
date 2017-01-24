# Handwritten Digit Recognition using Neural Networks

## Training Set

Handwritten Digits taken from [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

![Samples from the MNIST Database](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/mnistSet.png?raw=true)

## Architecture

![Neural Network with 2 Hidden Layers](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/TwoLayerNeuralNetwork.png?raw=true)

1. Input : handwritten vectorized image, 784 nodes.
2. Layer 1 : fully connected, 300 neurons. 235200 total weights. 300 biases. Sigmoid activation function.
3. Layer 2 : fully connected, 100 neurons. 30000 total weights. 100 biases. Sigmoid activation function.
4. Output layer : fully connected, 10 neurons. 1000 total weights. 10 biases. Sigmoid activation function.

total: 266610 total tunable parameters.

## Learning Method

Stochastic gradient descent backpropagation with Root Mean Square Propagation.

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

Hyperparameters: global learning rate (epsilon), decay rate (rho), number of iterations (Iters), stochastic sampling size.

1. for i = 1 ... Iters
2.  stochastically sample a minibatch of training images (N)
3.  compute the gradient of the cost with respect to the neural network parameters w. Use [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
4.  accumulate the squared gradient r with the decay rate. ![](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/rmsProp.gif?raw=true)
5.  compute the update paramater. ![](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/parameterUpdate.gif?raw=true)
6.  update the neural network parameters. ![](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/update.gif?raw=true)
7.  end