import gzip
import struct
import numpy as np

def read_images(file_path):
    """ Reads the MNIST image file and returns a numpy array of images. """

    with gzip.open(file_path, "rb") as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic_number != 2051:
            raise ValueError("Invalid magic number for image file")

        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)

        return images.astype(np.float32) / 255

def read_labels(file_path):
    """ Reads the MNIST label file and returns a numpy array of labels. """

    with gzip.open(file_path, "rb") as f:
        magic_number, _ = struct.unpack(">II", f.read(8))
        if magic_number != 2049:
            raise ValueError("Invalid magic number for image file")

        return np.frombuffer(f.read(), dtype=np.uint8)

def load_mnist_data(data_dir):
    """ Loads the MNIST data.

    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    train_images = read_images(f"{data_dir}/train-images-idx3-ubyte.gz")
    train_labels = read_labels(f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_images = read_images(f"{data_dir}/t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)
