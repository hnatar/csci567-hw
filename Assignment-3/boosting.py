import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T
    
        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        assert len(self.betas) == len(self.clfs_picked)
        r = np.zeros(shape=(1, len(features)) )
        for t in range(0, len(self.clfs_picked)):
            preds = np.array(self.clfs_picked[t].predict(features))
            r += self.betas[t] * preds
        r[r==0.0] = 1
        r[r<0.0] = -1
        r[r>0.0] = 1
        return r.astype(int)
        

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        sample_weights = [1.0/float(len(features))]*len(features)
        ht=None
        et=None
        for iterations in range(0, self.T):
            for classifier in self.clfs:
                val = 0
                predictions = classifier.predict(features)
                for i in range(0, len(predictions)):
                    if predictions[i] != labels[i]:
                        val += sample_weights[i]
                if et == None or val<et:
                    et = val
                    ht = classifier
            bt = 0.5*np.log( (1.0-et)/(et) )
            self.clfs_picked.append(ht)
            self.betas.append(bt)

            predictions = ht.predict(features)
            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    sample_weights[i] *= np.exp(-bt)
                else:
                    sample_weights[i] *= np.exp(bt)
            
            S = float(sum(sample_weights))
            for i in range(0, len(sample_weights)):
                sample_weights[i] = float(sample_weights[i])/S
        
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        pi = [0.5]*len(features)
        ft = [0]*len(features)
        sample_weights = [0]*len(features)
        for iterations in range(0, self.T):
            zt = [0]*len(features)
            for n in range(0, len(features)):
                zt[n] = ((labels[n]+1.0)/2.0 - pi[n])/(pi[n]*(1-pi[n]))
                sample_weights[n] = pi[n]*(1-pi[n])
            ht = None
            et = None
            for classifier in self.clfs:
                val = 0
                predictions = classifier.predict(features)
                for samp in range(0, len(features)):
                    val += sample_weights[samp]*(zt[samp]-predictions[samp])*(zt[samp]-predictions[samp])
                if et == None or val < et:
                    et = val
                    ht = classifier
            self.clfs_picked.append(ht)
            self.betas.append(0.5)
            predictions = ht.predict(features)
            for i in range(0, len(predictions)):
                ft[i] += 0.5 * predictions[i]
                pi[i] = 1.0 / (1+np.exp(-2.0*ft[i]))

        
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
    