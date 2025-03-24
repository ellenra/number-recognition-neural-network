import numpy as np

class Model(object):
    def __init__(self, layers):
        """
        Initializing layers, except the input layer, with random weights and biases.
        """
        self.layers = len(layers)
        self.layer_sizes = layers

        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        
    def _forward(self, input):
        activation = input
        for i, (biases, weights) in enumerate(zip(self.biases, self.weights)):
            activation = sigmoid(np.dot(weights, activation) + biases)
        return activation
    
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

