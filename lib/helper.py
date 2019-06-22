import numpy as np
import pickle
import tensorflow as tf
import os


def next_batch(X, y, batch_size, epochs_completed=0, index_in_epoch=0):
    """
    Return the next `batch_size` examples from this data set.

    :param X:
    :param y:
    :param batch_size:
    :param epochs_completed:
    :param index_in_epoch:
    :return:
    """
    start = index_in_epoch
    num_examples = len(X)

    # Shuffle for the first epoch
    if epochs_completed == 0 and start == 0:
        permutation = np.arange(num_examples)
        np.random.shuffle(permutation)
        X = X[permutation]
        y = y[permutation]

    # Go to the next epoch
    if start + batch_size > num_examples:
        # Finished epoch
        epochs_completed += 1

        # Get the rest examples in this epoch
        rest_num_examples = num_examples - start
        images_rest_part = X[start:num_examples]
        labels_rest_part = y[start:num_examples]

        # Shuffle the data
        permutation = np.arange(num_examples)
        np.random.shuffle(permutation)
        X = X[permutation]
        y = y[permutation]

        # Start next epoch
        start = 0
        _index_in_epoch = batch_size - rest_num_examples
        end = _index_in_epoch
        images_new_part = X[start:end]
        labels_new_part = y[start:end]

        return np.concatenate(
            (images_rest_part, images_new_part),
            axis=0
        ), np.concatenate(
            (labels_rest_part, labels_new_part),
            axis=0
        ), index_in_epoch
    else:
        index_in_epoch += batch_size
        end = index_in_epoch

        return X[start:end], y[start:end], index_in_epoch


def save_object(obj, filename):
    """

    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'rb') as input:
        return pickle.load(input)


def transform_mnist_data(x_train, y_train, x_test, y_test) -> (list, list, list, list, list, list):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:

    :return: x_train, y_train, x_test, y_test, x_validation, y_validation
    """
    X = np.concatenate((x_train, x_test), axis=0)
    X = np.array([[e / 256 for l in xs for e in l] for xs in X], dtype=np.float32)

    y = np.concatenate((y_train, y_test), axis=0)
    y = np.array([[1 if e == i else 0 for i in list(range(0, 10))] for e in y], dtype=np.float64)

    # test_and_validation_size
    s = round(len(X) * 0.1)

    return X[:-s * 2], y[:-s * 2], X[-s * 2:][:s], y[-s * 2:][:s], X[-s * 2:][:s], y[-s * 2:][:s]


def get_transformed_mnist_data(cached=True) -> (list, list, list, list, list, list):
    """
    cached get and transform mnist data set

    :return: x_test, y_test, x_train, y_train, x_validation, y_validation
    """
    files = ['x_train', 'y_train', 'x_test', 'y_test', 'x_validation', 'y_validation']
    files_exists = all([os.path.isfile('./.tmp/%s' % f) for f in files])

    if cached and files_exists:
        return tuple([load_object('./.tmp/%s' % f) for f in files])
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train, y_train, x_test, y_test, x_validation, y_validation = transform_mnist_data(
            x_train, y_train, x_test, y_test
        )

        save_object(x_train, './.tmp/x_train')
        save_object(y_train, './.tmp/y_train')
        save_object(x_test, './.tmp/x_test')
        save_object(y_test, './.tmp/y_test')
        save_object(x_validation, './.tmp/x_validation')
        save_object(y_validation, './.tmp/y_validation')

        return x_test, y_test, x_train, y_train, x_validation, y_validation
