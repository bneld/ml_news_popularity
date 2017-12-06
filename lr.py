import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn import preprocessing

class JN_Lasso:
    def __init__(self, alpha, max_iter=1000) :
        # hyperparameters
        self.alpha = alpha
        self.max_iter = max_iter
        # parameters
        self.betas = None
        self.beta_mins = None
        self.beta_maxes = None
        # data
        self.training = None
        self.norm_targets = None
        self.targets = None
        self.min_norm = 0
        self.max_target = None
        self.std_scaler = None

    def fit(self, data, targets):
        self.training = data
        self.targets = targets

        self.std_scaler = preprocessing.StandardScaler().fit(data)
        data = self.std_scaler.transform(data)
        targets = preprocessing.scale(targets, with_mean=True, with_std=False)

        # self.max_target = np.max(targets)
        # self.beta_mins = [np.min(data[:,j]) for j in range(len(data[0]))]
        # # print(self.beta_mins)
        # self.beta_maxes = [np.max(data[:,j]) for j in range(len(data[0]))]
        # # print(self.beta_maxes)
        # self.norm_targets = self.normalize_zero_max(self.targets, self.max_target)

        # norm = np.zeros((len(data), len(data[0])))
        # # normalize features
        # for j in range(len(data[0])):
        #     norm[:,j] = self.normalize(data[:,j], self.beta_mins[j], self.beta_maxes[j])
        
        # initialize weights
        self.betas = np.zeros(len(data[0]))
        # norm = normalize(self.training)
        # norm = self.training
        # iterate till not converged
        for iter in range(self.max_iter):
            # print("=== Iter {}".format(iter))
            # iterate over all features (j=0,1...M)            
            for j in range(len(self.betas)):
                # remove feature j to determine effect on residuals
                # b_no_j = self.betas.copy()
                # b_no_j[j] = 0
                # x_no_j = self.training
                self.betas[j] = self.update_beta(data, targets, self.betas, j)
                # x_no_j = np.delete(self.training, j, 1)
                # b_no_j = np.delete(self.betas, j, 0)
                # # predict_no_j = np.dot(b_no_j, x_no_j.T)
                # predict_no_j = x_no_j.dot(b_no_j)

                # # use normalized targets
                # residuals = self.norm_targets - predict_no_j
                # p_j = np.dot(norm[:,j], residuals)
                # print("Max Residual = {}".format(np.max(residuals)))
                # print(p_j)
                # # update beta_j based on residuals
                # self.betas[j] = self.soft_threshold(p_j, self.alpha / 2);
            # if iter % 100 == 0:
                # print(self.betas)
                
            # print(self.betas)
    def update_beta(self, data, targets, betas, j):
        n = len(targets)
        x_no_j = np.delete(data, j, 1)
        b_no_j = np.delete(betas, j, 0)
        norm_x_j = np.linalg.norm(data[:, j])
        # predict_no_j = np.dot(b_no_j, x_no_j.T)
        predicted = x_no_j.dot(b_no_j)

        # use normalized targets
        residuals = targets - predicted
        # p_j = data[:,j].dot(residuals)
        z_j = norm_x_j.dot(residuals)/n
        # print("Max Residual = {}".format(np.max(residuals)))
        # print(p_j)
        # update beta_j based on residuals

        # return self.soft_threshold(p_j, self.alpha*n/2)/(norm_x_j**2)
        return self.soft_threshold(z_j, self.alpha)


    def soft_threshold(self, p_j, threshold):
        if p_j < -threshold:
            return p_j + threshold
        elif p_j > threshold:
            return p_j - threshold
        else:
            return 0

    def predict(self, data):
        data = self.std_scaler.transform(data)
        predicted = data.dot(self.betas)
        return predicted

        # norm_data = np.zeros((len(data), len(data[0])))
        # # normalize features
        # for j in range(len(data[0])):
        #     norm_data[:,j] = self.normalize(data[:,j], self.beta_mins[j], self.beta_maxes[j])

        # norm_predicted = np.dot(self.betas, norm_data.T)
        # return self.denormalize_zero_max(norm_predicted, self.max_target)

    # def mse(self):
    #     return mean_squared_error(self.targets, self.predicted)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)
    def normalize_zero_max(self, arr, max_val):
        return arr / max_val
    def denormalize_zero_max(self, norm_arr, max_val):
        return norm_arr * max_val

    def loss(self):
        return np.sum(np.square(self.targets - self.predicted))\
            + self.alpha * np.sum(np.abs(self.betas))

    def plot(self):
        lw = 2
        plt.scatter(range(len(self.targets)), self.targets, color='darkorange', label='data')
        plt.scatter(range(len(self.predicted)), self.predicted, color='cornflowerblue', lw=lw, label='Polynomial model')
        plt.xlabel('Data')
        plt.ylabel('Target')
        plt.title('Lasso Regression')
        plt.legend()
        plt.show()
