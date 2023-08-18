# Flexible Neural Network from Scratch

## Description

This project implements a multi-layer neural network from scratch, using only foundational Python libraries such as NumPy. The architecture is crafted with flexibility in mind, enabling users to adjust the number of layers and neurons within those layers with ease.

## Features

- **Object-Oriented Design**: Utilizes an Object-Oriented Programming (OOP) approach for clarity and modularity.
  
- **Adjustable Architecture**: Easily specify any number of layers and neurons within each layer.
  
- **Activation Function**: Uses the sigmoid activation function for neurons.
  
- **Training Algorithms**: Implements both feedforward and backpropagation algorithms to train the neural network.

- **Versatility**: Suitable for various neural network applications, from basic XOR problems to more intricate datasets.

## Usage

1. Instantiate the `NeuralNetwork` class with your desired architecture. For instance, `[2, 4, 3, 1]` represents a network with:
   - 2 input neurons
   - First hidden layer: 4 neurons
   - Second hidden layer: 3 neurons
   - 1 output neuron

   ```python
   nn = NeuralNetwork([2, 4, 3, 1])

2. With your input data X and labels y defined, train your model:

  ```python
  nn.train(X, y, epochs=1000, learning_rate=0.1)

