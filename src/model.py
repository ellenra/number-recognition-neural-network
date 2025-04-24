import numpy as np
from pathlib import Path

class Model():
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        """ Initializes model with random weights and zero biases. """

        self.layers = 3
        self.accuracy = 0

        self.w1 = (np.random.randn(hidden_layer_size, input_layer_size)
                   * np.sqrt(2 / input_layer_size))
        self.w2 = (np.random.randn(output_layer_size, hidden_layer_size)
                   * np.sqrt(2 / hidden_layer_size))

        self.b1 = np.zeros((hidden_layer_size, 1))
        self.b2 = np.zeros((output_layer_size, 1))

    def forward(self, vector):
        """ Forward pass to calculate activations for layers. Sigmoid function
        for hidden layer and softmax for output layer.

        Args:
            vector: Numpy array, shape (784, 1)

        Returns:
            List of activation vectors for layers and list of z vectors = weighted inputs for layers
        """

        z1 = np.dot(self.w1, vector) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(self.w2, a1) + self.b2
        a2 = softmax(z2)

        return [vector, a1, a2], [z1, z2]

    def train_model(self, training_data, epochs, batch_size, learning_rate, test_data, save_model):
        """ Divides data to smaller batches, trains model and shows results.

        Args:
            training_data: List of tuples (vector, label)
            epochs: Number of epochs to train (epoch = one complete pass through the dataset)
            batch_size: Size of the mini batches
            learning_rate: Learning rate for the model
            test_data: List of tuples (vector, label)
        """

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            batches = []

            for i in range(0, len(training_data), batch_size):
                batches.append(training_data[i : i + batch_size])

            for batch in batches:
                nabla_b = [np.zeros_like(self.b1), np.zeros_like(self.b2)]
                nabla_w = [np.zeros_like(self.w1), np.zeros_like(self.w2)]

                for vector, label in batch:
                    gradients_for_biases, gradients_for_weights = self.backpropagate(vector, label)
                    for x, y in enumerate(gradients_for_biases):
                        nabla_b[x] += y
                    for x, y in enumerate(gradients_for_weights):
                        nabla_w[x] += y
                self.update_params(nabla_b, nabla_w, learning_rate, batch)

            self.accuracy = self.evaluate(test_data)
            print(f"{epoch}: {self.accuracy}% accuracy")
            
        if save_model:
            self.save_trained_model()

    def update_params(self, nabla_b, nabla_w, learning_rate, batch):
        self.w1 -= (learning_rate / len(batch)) * nabla_w[0]
        self.w2 -= (learning_rate / len(batch)) * nabla_w[1]
        self.b1 -= (learning_rate / len(batch)) * nabla_b[0]
        self.b2 -= (learning_rate / len(batch)) * nabla_b[1]

    def backpropagate(self, vector, label):
        """ Calculates the gradients of weights and biases.

        Args:
            vector: Column vector of one number
            label: Correct number

        Returns:
            Gradients of biases and gradients of weights
        """

        activations, z_values = self.forward(vector)

        output_error = activations[-1] - label

        output_layer_weight_gradients = np.dot(output_error, activations[-2].T)
        output_layer_weight_biases = np.sum(output_error, axis=1, keepdims=True)

        hidden_error = np.dot(self.w2.T, output_error) * sigmoid_prime(z_values[-2])

        hidden_layer_weight_gradients = np.dot(hidden_error, activations[0].T)
        hidden_layer_weight_biases = np.sum(hidden_error, axis=1, keepdims=True)

        return ([hidden_layer_weight_biases, output_layer_weight_biases],
                [hidden_layer_weight_gradients, output_layer_weight_gradients])

    def evaluate(self, test_data):
        """ Evaluates performance.

        Returns:
            Accuracy of model as a percentage
        """
        results = [(np.argmax(self.forward(x)[0][-1]), np.argmax(y)) for (x, y) in test_data]
        accuracy = sum(int(x == y) for (x, y) in results) / len(test_data)
        accuracy_percentage = round(accuracy * 100, 4)
        return accuracy_percentage

    def save_trained_model(self):
        directory = Path("files")
        directory.mkdir(parents=True, exist_ok=True)
        
        path = directory / f"accuracy{round(self.accuracy, 4)}.npz"
        
        np.savez(path, w1=self.w1, w2=self.w2, b1=self.b1, b2=self.b2)

    def load_latest_model(self, folder="files"):
        dir = Path(folder)
        files = list(dir.glob("*.npz"))
        if not files:
            return

        newest_file = max(files, key=lambda f: f.stat().st_mtime)
        data = np.load(newest_file)

        self.w1 = data["w1"]
        self.w2 = data["w2"]
        self.b1 = data["b1"]
        self.b2 = data["b2"]        


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """ Derivative of the sigmoid function. """
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0)

