import numpy as np
import pandas as pd
from collections import defaultdict
import re
import nltkfrom nltk.corpus
import stopwords set(stopwords.words('english'))

class NaiveBayes:
    def __init__(self):
        self.classes = categories


    def train(self, X, y):
        self.classes = np.unique(y)
        self.num_features = X.getrow(0).shape[1]
        self.theta_k = [] #marginal probability for each class
        self.theta_jk = [] #confitional probability of each feature given each class [each class[each features]]

        for k in self.classes:
            t_k = y[y == k].count()/ y.count() # (# of examples where y=1) / (total # of examples)
            X_k = X.toarray()[y[y == k].index]
            self.theta_k.append(t_k)
            t_jk = []
            for i in range(self.num_features):
                tjk = sum(X_k[:, i]) /y[y == k].count() #(# of examples where xj=1 and y=k) / (# of examples where y=k)
                t_jk.append(tjk)

            self.theta_jk.append(t_jk)

    def predict(self, x):
        class_prob = []
        for k in range(self.classes):
            feature_likelihood = 0
            for j in range(num_features):
                feature_likelihood += x[j]*np.log(theta_jk[k][j]+(1-x[j])*np.log(1-theta_jk[k][j]))
            cb = feature_likelihood+np.log(theta_k[k])
            class_prob.append(cb)
        return np.argmax(class_prob)
