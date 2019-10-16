import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn import preprocessing



class NaiveBayes:
    def __init__(self):
        self.classes = []
        self.num_features = 0
        self.theta_k = []
        # self.theta_jk = []
        prediction = []
        self.encoder = preprocessing.LabelEncoder()

    def train(self, X, y):
        self.classes = np.unique(y)
        self.encoder.fit(y)
        self.num_features = X.getrow(0).shape[1]
        self.theta_k = [] #marginal probability for each class
        self.theta_jk = np.empty((len(self.encoder.classes_), X.shape[1])) #confitional probability of each feature given each class [each class[each features]]
        print(self.theta_jk.shape)
        # X = X.toarray()
        print(X, X.shape)
        y = np.array(y)
        print(self.encoder.classes_)
        print(y)
        # X_k = np.array()

        num_examples = len(y)
        for i, k in enumerate(self.encoder.classes_):
            # print(len(y[y == k]))
            length_y = len(y[y == k])
            t_k = length_y / num_examples # (# of examples where y=1) / (total # of examples)
            X_k = X[np.where(y == k)]
            print(X_k.shape)
            #X_k = X.toarray()[y[y == k].index]
            self.theta_k.append(t_k)
            sum_array = np.sum(X_k, axis=0)
            print(sum_array, sum_array.shape)
            t_jk = (sum_array + 1) / (length_y + 2)
            print(t_jk, t_jk.shape)
            # for i in range(self.num_features):
            #     # print(sum_array[i], len(y[y == k]))
            #     tjk = (sum_array[i]+1) /(length_y+2) #(# of examples where xj=1 and y=k) / (# of examples where y=k)
            #     t_jk.append(tjk)
            print('nested loop done')
            # A = numpy.vstack([A, newrow])
            self.theta_jk[i] = t_jk

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
