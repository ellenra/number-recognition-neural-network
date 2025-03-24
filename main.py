from src.data_loader import load_mnist_data
from src.model import Model

def main():
    (images, labels), (test_images, test_labels) = load_mnist_data()
    
    """
    Network has three layers: input layer has 784 neurons, hidden layer has 128 and output layer has 10.
    """
    model = Model([784, 128, 10])

    input_image = images[:, 0].reshape(-1, 1)
    
    output = model._forward(input_image)
    
    print(output)

if __name__ == "__main__":
    main()
