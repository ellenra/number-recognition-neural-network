import gzip
import numpy as np
import os

def read_images(file_path):
    with gzip.open(file_path) as f:
        if int.from_bytes(f.read(4), byteorder='big') != 2051:
            raise ValueError("Invalid magic number for image file")

        images_amount = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')

        buffer = f.read(images_amount * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255
        return images.reshape(images_amount, rows * cols).T

def read_labels(file_path):
    with gzip.open(file_path) as f:
        if int.from_bytes(f.read(4), byteorder='big') != 2049:
            raise ValueError("Invalid magic number for label file")

        labels_amount = int.from_bytes(f.read(4), byteorder='big')
        return np.frombuffer(f.read(labels_amount), dtype=np.uint8)

def load_mnist_data():
    images = read_images("mnist_data/train-images-idx3-ubyte.gz")
    labels = read_labels("mnist_data/train-labels-idx1-ubyte.gz")
    test_images = read_images("mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = read_labels("mnist_data/t10k-labels-idx1-ubyte.gz")

    return (images, labels), (test_images, test_labels)
