# Handwritten Digit Recognition using Neural Networks

## Training Set

Handwritten Digits taken from [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

![Samples from the MNIST Database](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/mnistSet.png?raw=true)

## Architecture

![Neural Network with 2 Hidden Layers](https://github.com/Gregjksmith/Handwritten-Digit-Recognition-Neural-Network/blob/master/pictures/TwoLayerNeuralNetwork.png?raw=true)

Input : handwritten vectorized image, 784 nodes.
Layer 1 : fully connected, 300 nodes. 235200 total weights. 300 biases. Sigmoid activation function.
Layer 2 : fully connected, 100 nodes. 30000 total weights. 100 biases. Sigmoid activation function.
Output layer : fully connected, 10 nodes. 1000 total weights. 10 biases. Sigmoid activation function.

total: 266610 total tunable parameters.

## Learning Method

stochastic gradient descent backpropagation with RMSProp adaptive learning rates.

### Cost Function

Given a set of training vectors {xn} where n = 1...N, with corresponding target vectors {tn}, we seek to minimize the cost function E(w).

$ E(w)=\frac{1}{N}\sum_{n=1}^{N}\left \| y(x_{n},w)-t_{n} \right \|^{2}