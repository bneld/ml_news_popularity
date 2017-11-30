import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class JN_Lasso:
    def __init__(self, alpha) :
        self.alpha = alpha
        self.lassoReg = Lasso(alpha=alpha)
        self.predicted = None

    def fit(self, data, targets):
        self.training = data
        self.targets = targets
        self.lassoReg.fit(data, targets)

    def predict(self, data):
        self.predicted = self.lassoReg.predict(data)
        return self.predicted

    def mse(self):
        return mean_squared_error(self.targets, self.predicted)

    def plot(self):
        lw = 2
        plt.scatter(range(len(self.targets)), self.targets, color='darkorange', label='data')
        plt.scatter(range(len(self.predicted)), self.predicted, color='cornflowerblue', lw=lw, label='Polynomial model')
        plt.xlabel('Data')
        plt.ylabel('Target')
        plt.title('Lasso Regression')
        plt.legend()
        plt.show()
    