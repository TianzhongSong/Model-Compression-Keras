from keras.datasets import cifar10, cifar100
import numpy as np


def load_data(data='c10'):
    if data == 'c10':
        print('Loading CIFAR-10 dataset')
        nb_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std
    else:
        print('Loading CIFAR-100 dataset')
        nb_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        mean = np.array([129.3, 124.1, 112.4])
        std = np.array([68.2, 65.4, 70.4])
        x_train -= mean
        x_train /= std
        x_test -= mean
        x_test /= std

    return x_train, y_train, x_test, y_test, nb_classes
