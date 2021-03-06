from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.trainX = None
        self.trainLabel = None

    def train(self, features: List[List[float]], labels: List[int]):
        """ just carry dataset """
        self.trainX = features
        self.trainLabel = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        Predictions = []
        for test in features:
            """ each pair is (distance to example, label of example) """
            Pairs = []
            for i, example in enumerate(self.trainX):
                Pairs.append( (self.distance_function(test, example), self.trainLabel[i],) )
            """ sort pairs by distance to example, so closest examples appear first.
                then, 'trim' to keep only first k examples for kNN. Then find mode / most frequent label """
            Pairs = sorted(Pairs, key = lambda x: x[0])[: self.k+1]
            PredictLabels = [y for (x,y) in Pairs]
            Predictions.append( scipy.stats.mode(PredictLabels).mode[0] )
        return Predictions

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
