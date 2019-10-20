import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn import preprocessing

class NaiveBayes:
    def __init__(self):
        self.classes = []
        self.num_features = 0
        self.theta_k = []
        self.encoder = preprocessing.LabelEncoder()

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.encoder.fit(y)
        self.num_features = X.getrow(0).shape[1]
        self.theta_k = [] #marginal probability for each class
        self.theta_jk = np.empty((len(self.encoder.classes_), X.shape[1])) #confitional probability of each feature given each class [each class[each features]]
        print(self.theta_jk.shape)
        y = np.array(y)

        num_examples = len(y)
        for i, k in enumerate(self.encoder.classes_):
            length_y = len(y[y == k])
            t_k = length_y / num_examples # (# of examples where y=1) / (total # of examples)
            X_k = X[np.where(y == k)]
            self.theta_k.append(t_k)
            sum_array = np.sum(X_k, axis=0)
            t_jk = (sum_array + 1) / (length_y + 2)
            self.theta_jk[i] = t_jk
        return self
   
    def predict(self, X, y=None):
        X = X.toarray()
        print("predict start")
        print(len(X[0]), self.num_features, len(self.theta_jk[0]))
        prediction = []
        for i in range(len(X)):
            class_prob = []
            for k in range(len(self.classes)):                
                feature_likelihood = np.dot(X[i],np.log(self.theta_jk[k])) + np.dot((1-X[i]), np.log(1-self.theta_jk[k]))              
                cb = feature_likelihood+np.log(self.theta_k[k])
                class_prob.append(cb)
            prediction.append(np.argmax(class_prob))
        return prediction

    def score(self, X, y=None):
        prediction = self.predict(X)
        correct = sum(np.equal(self.classes[prediction], y))
        return correct / len(prediction)

    # def validation(self, prediction, y):
    #     # print("lenth of prediction ", self.classes[prediction])
    #     # print("lenth of y ", y)
    #     correct = sum(np.equal(self.classes[prediction], y))
    # #    c = np.logical_and(prediction == y)
    #     # for i in range(len(prediction)):
    #     #     correct += 1 if y[prediction[i]] == y[i] else 0
    #     accuracy = correct / len(prediction)
    #     return accuracy
