from src.data_loader import load_mnist_data
from src.model import Model
import numpy as np

def main():
    """
    This is run to train the model.

    Steps:
    1. Loads the MNIST training and testing data.
    2. Reshapes images from 28x28 into a 784x1 column vector.
    3. One-hot encodes correct labels (converts into a 10x1 column vector) to match the
    format the neural network's output format.
    4. Zips the data.
    5. Initializes a model with layer sizes.
    6. Loads pre-trained model if available.
    7. Trains the model using given settings.
    """
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data("mnist_data")

    train_images = [image.reshape(784, 1) for image in train_images]
    test_images = [image.reshape(784, 1) for image in test_images]
    
    train_labels = [np.eye(10)[label].reshape(10, 1) for label in train_labels]
    test_labels = [np.eye(10)[label].reshape(10, 1) for label in test_labels]

    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))

    model = Model(784, 128, 10)
    model.load_latest_model()
    model.train_model(training_data, epochs=20, batch_size=10, learning_rate=3.0, test_data=test_data, save_model=True)

if __name__ == "__main__":
    main()
