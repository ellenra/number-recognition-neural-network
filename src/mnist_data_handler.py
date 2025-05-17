import gzip
import struct
import numpy as np

def read_images(file_path):
    """ Reads the MNIST image file and returns a numpy array of images.
    """
    with gzip.open(file_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        images = images.astype(np.float32) / 255

        return images

def read_labels(file_path):
    """ Reads the MNIST label file and returns a numpy array of labels.
    """
    with gzip.open(file_path, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
def reshape_mnist_data(train_images, train_labels, test_images, test_labels):
    """ 1. Reshapes images from 28x28 into a 784x1 column vector.
    2. One-hot encodes correct labels (converts into a 10x1 column vector) to match the
    format the neural network outputs.
    3. Zips the data.
    """
    train_images = [image.reshape(784, 1) for image in train_images]
    test_images = [image.reshape(784, 1) for image in test_images]
    
    train_labels = [np.eye(10)[label].reshape(10, 1) for label in train_labels]
    test_labels = [np.eye(10)[label].reshape(10, 1) for label in test_labels]

    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))
    
    return training_data, test_data

def load_mnist_data(data_dir):
    """ Loads, reshapes and returns the MNIST data for the model. 
    """
    train_images = read_images(f"{data_dir}/train-images-idx3-ubyte.gz")
    train_labels = read_labels(f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_images = read_images(f"{data_dir}/t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    
    return reshape_mnist_data(train_images, train_labels, test_images, test_labels)
