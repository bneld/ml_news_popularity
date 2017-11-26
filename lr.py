import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

class JN_Lasso:
    def __init__(self, alpha) :
        self.alpha = alpha
        self.lassoReg = Lasso(alpha=alpha)
        self.predicted = None

    def fit(self, data, target):
        self.training = data
        self.target = target
        self.lassoReg.fit(data, target)

    def predict(self, data):
        self.predicted = self.lassoReg.predict(data)
        return self.predicted

    def plot(self):
        lw = 2
        plt.scatter(*range(len(self.target)), self.target, color='darkorange', label='data')
        plt.scatter(*range(len(self.predicted)), self.predicted, color='cornflowerblue', lw=lw, label='Polynomial model')
        plt.xlabel('Data')
        plt.ylabel('Target')
        plt.title('Lasso Regression')
        plt.legend()
        plt.show()