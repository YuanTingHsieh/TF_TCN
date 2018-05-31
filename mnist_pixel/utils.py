from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def data_generator(data_path, permuted = False):
    """
    Generate data for MNIST
    """
    mnist = input_data.read_data_sets(data_path, one_hot = True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels
    Y_test = mnist.test.labels

    mean = 0.1307
    std = 0.3081

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, Y_train, X_test, Y_test