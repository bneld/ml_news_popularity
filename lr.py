import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
lassoReg = Lasso(alpha=0.3, normalize=True).fit(X, y)
predicted = lassoReg.predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, predicted, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


#SVR implementation : using polynomail and RBF kernals
#sample data
#parameters: 
# from __future__ import division
# from numpy.linalg import inv

# k = 0 # kernal type : 0 polynomail , 1 : RBF
# c = 0 # regularization parameter (penalty parameter) 
# gamma = 0 # kernel coffecient
# degree = 2 # 
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()


# def calc_K_Mat(input_mat, degree, c ) : 
#     k_mat = np.ones([input_mat.shape[0]+1 ,input_mat.shape[0]+1 ])
#     k_mat[0][0] = 0
#     i_matrix = np.identity(input_mat.shape[0])
#     for i in range(1, k_mat.shape[0]) : 
#         for j in range (1, k_mat.shape[0]) : 
#             #Kernal 
#             k = (np.sum((input_mat[i-1:i , : ]).T * input_mat[j-1:j , :]) ** degree)  + (1/c) * i_matrix[i-1][j-1]
# #             k = ((np.sum((input_mat[i-1:i , : ]).T * input_mat[j-1:j , :]))  ** degree)  
#             k_mat[i][j] = k
#     return k_mat
    
# def fit(input_data, target , degree , c ) : #Moore-Penrose Pseudo-Inverse Matrix
#     k_matrix = calc_K_Mat(input_data ,degree , c)
#     target = np.insert(target, 0 , 0)
#     print("Target : ", target.shape)
#     print('The K :' , k_matrix.shape)
# #     print(k_matrix)
#     #get them w's 
#     p1 = inv(np.matmul(k_matrix.T , k_matrix ))
#     print("P1 : ", p1.shape)
#     p2 = np.matmul( k_matrix.T , target)
#     print("P2 : ", p2.shape)
#     alphas = np.matmul(p1 ,p2 )
#     print("alpha : ", alphas.shape)
#     return alphas

# def predict(input_data, alpha) : 
#     k_mat= calc_K_Mat(input_data , 2 ,7000 ) 
#     print("SUM : " , np.sum(alpha[1:]))
# #     return np.matmul(input_data.T , alpha[1:])
#     return np.matmul(alpha[1:].T , k_mat[1: , 1:]) + alpha[0]


# #TIME FOR TEST THO
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
# # y[::5] += 3 * (0.5 - np.random.rand(8))

# w = fit(X, y , 2 , 7000)
# print(w)

# y_p = predict(X, w)
# lw = 2


# plt.scatter(X, y, color='darkorange', label='data')
# # plt.plot(X, y_p[1:], color='red', lw=lw, label='Polynomial model')
# # plt.plot(X, y_p[1:], color='red', lw=lw, label='Polynomial model')
# plt.plot(X, y_p, color='red', label='Polynomial model')

# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()