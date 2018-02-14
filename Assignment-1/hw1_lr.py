from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        """ REMOVE THIS BEFORE SUBMIT """
        numpy.set_printoptions(precision=2)
        self.nb_features = nb_features
        """ Stores feature weights + bias as
            column vector after training """
        self.weights = None

    def include_bias_weight(self, features: List[List[float]]) -> List[List[float]]:
        """ Prepend each feature vector with a 
            1 to account for bias in model """
        feat_with_bias = []
        for feature in features:
            assert len(feature) == self.nb_features
            feat_with_bias.append( [1.0] + feature )
        return feat_with_bias

    def train(self, features: List[List[float]], values: List[float]):
        feat_with_bias = self.include_bias_weight(features)
        """ X = N x nb_features """
        X = numpy.matrix(feat_with_bias)
        print('X=\n', X)
        """ Y = N x 1 """
        """ (transpose to convert from row vector -> col. vector) """
        Y = numpy.matrix(values).transpose()
        print('Y=\n', Y)
        """ Xt = nb_features x N """
        Xt = X.transpose()
        """ Xt X = nb_features x nb_features """
        XtX = numpy.matmul(Xt, X)
        """ XtXi = nb_features x nb_features """
        """ numpy will find an 'inverse' even if the matrix
            is singular """
        XtXi = numpy.linalg.inv(XtX)
        """ weights = (Xt X)-1 Xt Y
                    = (nb_features x nb_features) x (nb_features x N x N x 1)
                    = (nb_features x nb_features) x (nb_features x 1)
                    = (nb_features x 1) """
        self.weights = numpy.matmul(XtXi, numpy.matmul(Xt, Y))
        print('Weights=\n', self.weights)

    def predict(self, features: List[List[float]]) -> List[float]:
        """" X = N x nb_features """
        feat_with_bias = self.include_bias_weight(features)
        X = numpy.matrix(feat_with_bias).transpose()
        """ W = nb_features x 1 """
        Wt = self.weights.transpose()
        Y = numpy.matmul(Wt, X)
        return numpy.asarray(Y).squeeze()

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.astype(float)


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
