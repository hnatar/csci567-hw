from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=100, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.norm_w = 0
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        current_iteration = 0
        while current_iteration < self.max_iteration:
            """ Perform one more iteration through training set and update weights """
            misclassified = 0
            
            for i, xi in enumerate(features):
                yi = labels[i]
                wTx = np.dot(self.w, xi)
                if yi*wTx <= 0:
                    misclassified += 1
                    xi_norm = np.linalg.norm(xi)
                    if np.absolute(xi_norm) < 0.000001:
                        """ smallest epsilon for 6 digits of precision """
                        xi_norm = 0.000001
                    self.w += yi * (xi / xi_norm)

            if misclassified == 0:
                """ no point was misclassified, so current weights will not be updated """
                return True
            current_iteration += 1
        return False




    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        labels = []
        for xi in features:
            """ yi = wT x """
            if np.dot(self.w, xi) >= 0:
                labels.append(1)
            else:
                labels.append(-1)
        return labels


    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    