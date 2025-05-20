import numpy as np
import pickle
import os

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        Y = batch[b'labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to (N, 32, 32, 3)
        Y = np.array(Y)
        return X, Y

def load_dataset():
    """
    Loads the CIFAR-10 dataset from local files and returns binary-labeled cat vs. non-cat data.
    Each image is labeled as a cat (1) or non-cat (0).
    """

    data_dir = 'cifar-10-batches-py'
    cat_class_index = 3  # CIFAR-10: 3 is 'cat'

    # Load training data
    X_train_list, Y_train_list = [], []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        X_batch, Y_batch = load_cifar10_batch(file_path)
        X_train_list.append(X_batch)
        Y_train_list.append(Y_batch)
    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)

    # Load test data
    X_test, Y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    # Convert to binary labels: 1 for cat, 0 for non-cat
    Y_train_binary = (Y_train == cat_class_index).astype(int).reshape(1, -1)
    Y_test_binary = (Y_test == cat_class_index).astype(int).reshape(1, -1)

    # Simulate class labels
    classes = np.array([b'non-cat', b'cat'])

    return X_train, Y_train_binary, X_test, Y_test_binary, classes
