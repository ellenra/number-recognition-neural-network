import unittest
import numpy as np
from src.model import Model
from src.mnist_data_handler import load_mnist_data

def mock_mnist_data():
    (_, _), (test_images, test_labels) = load_mnist_data("mnist_data")

    data_for_tests = np.random.choice(len(test_images), 10, replace=False)

    test_images = [test_images[img] for img in data_for_tests]
    test_labels = [test_labels[label] for label in data_for_tests]

    test_images = [img.reshape(784, 1) for img in test_images]
    test_labels = [np.eye(10)[label].reshape(10, 1) for label in test_labels]

    return test_images, test_labels

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(784, 128, 10)
        test_images, test_labels = mock_mnist_data()
        self.test_data = list(zip(test_images, test_labels))

    def test_forward_pass(self):
        vector, _ = self.test_data[0]
        activations, z_values = self.model.forward(vector)
        self.assertEqual(len(activations), 3)
        self.assertEqual(len(z_values), 2)
        self.assertEqual(len(activations[-1]), 10)

    def test_training(self):
        try:
            self.model.train_model(self.test_data, 1, 10, 3.0, self.test_data, save_model=False)
        except:
            print("Error in training!")

    def test_weights_and_biases_change(self):
        w1 = self.model.w1.copy()
        w2 = self.model.w2.copy()
        b1 = self.model.b1.copy()
        b2 = self.model.b2.copy()

        self.model.train_model(self.test_data, 1, 10, 3.0, self.test_data, save_model=False)

        self.assertFalse(np.array_equal(w1, self.model.w1))
        self.assertFalse(np.array_equal(w2, self.model.w2))
        self.assertFalse(np.array_equal(b1, self.model.b1))
        self.assertFalse(np.array_equal(b2, self.model.b2))
        
    def test_accuracy_increases(self):
        before = self.model.evaluate(self.test_data)
        self.model.train_model(self.test_data, 5, 10, 3.0, self.test_data, save_model=False)
        after = self.model.evaluate(self.test_data)
        
        self.assertGreaterEqual(after, before)
