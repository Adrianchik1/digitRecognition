from .MnistDataloader import MnistDataloaderClass
import random
import matplotlib.pyplot as plt
from os.path  import join
import numpy as np

def loadData2():
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = 'C:/Users/adria/source/repos/digitRecognition/MNISTdata/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloaderClass(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.loadMnistData()

    X_train = np.array(x_train)  # shape (60000, 28, 28)
    y_train = np.array(y_train)  # shape (60000,)

    X_test = np.array(x_test)
    y_test = np.array(y_test)
     
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train, y_train = reshapeData(X_train, y_train)
    return X_train, y_train

def reshapeData(X_train, y_train):
    y = []
    for label in y_train:
        singleY = [0] * 10
        singleY[label] = 1
        y.append(singleY)

    X_train = X_train.reshape(60000, 784)

    return X_train, y 