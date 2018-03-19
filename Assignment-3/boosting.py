import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
    def __init__(self, clfs: List[Classifier], T=0):
        self.clfs = list(clfs)
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
        r = [0]*len(features)

        for t in range(0, len(self.clfs_picked)):
            preds = self.clfs_picked[t].predict(features)
            print('preds=',preds)
            for n in range(0, len(features)):
                r[n] += self.betas[t] * preds[n]

        for i in range(0, len(r)):
            if r[i]<0:
                r[i] = -1
            else:
                r[i] = 1
        return r
        

class AdaBoost(Boosting):
    def __init__(self, clfs: List[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        sample_weights = [1.0/float(len(features))]*len(features)
        ht=None
        et=None
        for iterations in range(0, self.T):
            for t in range(0, self.num_clf):
                val = 0
                predictions = self.clfs[t].predict(features)
                for i in range(0, len(predictions)):
                    if predictions[i] != labels[i]:
                        val += sample_weights[i]
                if et == None or val<et:
                    et = val
                    ht = t
            bt = 0.5*np.log( (1.0-et)/(et) )
            self.clfs_picked.append(self.clfs[ht])
            self.betas.append(bt)

            predictions = self.clfs[ht].predict(features)
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
    def __init__(self, clfs: List[Classifier], T=0):
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
                zt[n] = (0.5*(labels[n]+1.0) - pi[n])/(pi[n]*(1.0-pi[n]))
                sample_weights[n] = pi[n]*(1-pi[n])
            ht = None
            et = None
            for hyp in range(0, len(self.clfs)):
                val = 0
                predictions = self.clfs[hyp].predict(features)
                for samp in range(0, len(features)):
                    val += sample_weights[samp]*(zt[samp]-predictions[samp])*(zt[samp]-predictions[samp])
                if et == None or val <= et:
                    et = val
                    ht = hyp
            self.clfs_picked.append(self.clfs[ht])
            self.betas.append(0.5)
            predictions = self.clfs[ht].predict(features)
            for i in range(0, len(predictions)):
                ft[i] += 0.5 * predictions[i]
                pi[i] = 1.0 / (1+np.exp(-2.0*ft[i]))

        
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
    