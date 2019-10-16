import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack, csr_matrix


class NaiveBayes:
    def __init__(self):
        self.classes = []
        self.num_features = 0
        self.theta_k = []
        self.theta_jk = []
        prediction = []

    def train(self, X, y):
        self.classes = np.unique(y)
        self.num_features = X.getrow(0).shape[1]
        self.theta_k = [] #marginal probability for each class
        self.theta_jk = [] #confitional probability of each feature given each class [each class[each features]]

        for k in self.classes:
            t_k = y[y == k].count()/ y.count() # (# of examples where y=1) / (total # of examples)
            X_k = X.toarray()[np.where(np.array(y) == k)]
            #X_k = X.toarray()[y[y == k].index]
            self.theta_k.append(t_k)
            t_jk = []
            for i in range(self.num_features):
                tjk = (sum(X_k[:, i])+1) /(y[y == k].count()+2) #(# of examples where xj=1 and y=k) / (# of examples where y=k)
                t_jk.append(tjk)

            self.theta_jk.append(t_jk)

    def predict(self, X):
        prediction = []
        for i in range(len(X)):
            class_prob = []
            for k in range(len(self.classes)):
                feature_likelihood = 0
                for j in range(num_features):
                    feature_likelihood += x[j]*np.log(theta_jk[k][j]+(1-x[j])*np.log(1-theta_jk[k][j]))
                cb = feature_likelihood+np.log(theta_k[k])
                class_prob.append(cb)
            prediction.append(np.argmax(class_prob))
        return prediction

    def validation(self, prediction, y):
        correct = 0
        for i in range(len(prediction)):
            correct += 1 if np.unique(y)[prediction[i]] == y[i] else 0
        accuracy = correct / len(prediction)
        return accuracy
