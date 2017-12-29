import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn import preprocessing

class JN_Ridge:
    def __init__(self, alpha) :
        # hyperparameters
        self.betas = None
        # parameter
        self.alpha = alpha
        # data
        self.training = None
        self.targets = None
        

    def fit(self, X, Y):
        self.training = X
        self.targets = Y

        I = np.identity(X.shape[1])

        inverted = np.linalg.inv(np.matmul(X.T, X) + self.alpha*I)
        
        self.betas = np.matmul(np.matmul(inverted, X.T), Y)

    def predict(self, data):
        # data = self.std_scaler.transform(data)
        # predicted = data.dot(self.betas)
        # return predicted
        return data.dot(self.betas)

