from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    mse = np.sum( np.power(yt-yp, 2) )/ len(y_pred)
    return mse


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    precision = 0.0
    recall = 0.0
    for i, x in enumerate(predicted_labels):
        if x == real_labels[i]:
            precision += 1.0
    for i, x in enumerate(real_labels):
        if x == predicted_labels[i]:
            recall += 1.0
    precision = precision / len(predicted_labels)
    recall = recall / len(real_labels)
    if precision + recall == 0:
        return 0
    return 2.0 * precision * recall / (precision + recall)

def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    r = []
    for k in range(1, k+1):
        r.append( np.power(features, k).transpose() )
    return np.concatenate(r).transpose()


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    p1, p2 = np.array(point1), np.array(point2)
    d = np.dot(p1-p2, p1-p2)
    return np.sqrt(d)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    p1, p2 = np.array(point1), np.array(point2)
    return np.dot(p1, p2)


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    p1, p2 = np.array(point1), np.array(point2)
    d = np.dot(p1-p2, p1-p2)
    return -np.exp( -0.5 * np.sqrt(d*d) )


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """ Normalize and return """
        NormalizedVectors = []
        for vec in features:
            x = np.array(vec)
            N = float( np.sqrt(np.dot(x, x)) )
            if N == 0:
                NormalizedVectors.append(x)
            else:
                NormalizedVectors.append(x/N)
        return NormalizedVectors


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.initialized = False
        self.range_min = None
        self.diff = None
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        if self.initialized == False:
            """ Find max(x_i) - min(x_i) where x_i are the values
                appearing in the training set for the i-th dimension.
                To normalize, divide each dimension by corresponding value. """
            self.range_min = np.amin(features, axis=0)
            self.range_max = np.amax(features, axis=0)
            self.diff = self.range_max - self.range_min
            self.initialized = True
        """ Normalize and return """
        NormalizedVectors = []
        for vec in features:
            NormalizedVectors.append( (np.array(vec) - self.range_min)/self.diff )
        return NormalizedVectors


