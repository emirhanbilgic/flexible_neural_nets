import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

class NeuralNetwork:
    def __init__(self, layers_config):
        self.layers = []
        for i in range(len(layers_config) - 1):
            layer = Layer(layers_config[i], layers_config[i+1])
            self.layers.append(layer)
    
    def feedforward(self, X):
        self.a = [X]
        for layer in self.layers:
            z = np.dot(self.a[-1], layer.weights) + layer.biases
            activation = sigmoid(z)
            self.a.append(activation)
        return self.a[-1]
    
    def backpropagation(self, X, y, learning_rate):
        m = X.shape[0]
        dZ = self.a[-1] - y
        for i in reversed(range(len(self.layers))):
            dW = np.dot(self.a[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 0:
                dA_prev = np.dot(dZ, self.layers[i].weights.T)
                dZ = dA_prev * sigmoid_derivative(self.a[i])
            
            # Update weights and biases
            self.layers[i].weights -= learning_rate * dW
            self.layers[i].biases -= learning_rate * np.sum(db)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - self.a[-1]) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

#example:
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

#you can now define any number of layers and neurons per layer.
#example: 2 input neurons, 4 hidden neurons in the first hidden layer, 3 in the second, and 1 output neuron
nn = NeuralNetwork([2, 4, 3, 1])
nn.train(X, y, epochs=1000, learning_rate=0.1)
