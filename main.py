from src.data_loader import load_mnist_data
from src.model import Model, one_hot
import numpy as np

def main():
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data("mnist_data")

    train_images = [image.reshape(784, 1) for image in train_images]
    test_images = [image.reshape(784, 1) for image in test_images]
    
    train_labels = [np.eye(10)[label].reshape(10, 1) for label in train_labels]
    test_labels = [np.eye(10)[label].reshape(10, 1) for label in test_labels]

    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))

    model = Model(784, 128, 10)
    model.train_model(training_data, 30, 10, 3.0, test_data, save_model=False)

if __name__ == "__main__":
    main()
