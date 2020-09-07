import csv
import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return super().default(x)


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
    return x * (1 - x)


def softmax(a) -> np.ndarray:
    exp = np.exp(a - np.max(a, axis=0))
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(outputs, labels, epsilon=1e-12) -> np.double:
    # avoid error
    outputs = np.clip(outputs, epsilon, 1. - epsilon)
    return -np.sum(labels * np.log(outputs))


def load_csv_np(path):
    """
    load csv as np array
    :param path: file path
    """
    with open(path) as fd:
        return np.asarray([np.asarray(row, dtype=np.double) for row in csv.reader(fd)])


def load_csv(path):
    """
    load csv as list
    :param path: file path
    """
    with open(path) as fd:
        for row in csv.reader(fd):
            yield row
