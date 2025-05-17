from src.mnist_data_handler import load_mnist_data, reshape_mnist_data
from src.model import Model

def main():
    """
    This is run to train the model.

    Steps:
    1. Loads the MNIST training and testing data.
    2. Initializes a model with layer sizes.
    3. Loads pre-trained model if available.
    4. Trains the model using given settings.
    """
    training_data, test_data = load_mnist_data("mnist_data")

    model = Model(784, 128, 10)
    model.load_latest_model()
    model.train_model(training_data, epochs=20, batch_size=10, learning_rate=3.0, test_data=test_data, save_model=True)

if __name__ == "__main__":
    main()
