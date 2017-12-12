import numpy as np
from numpy.linalg import inv
class JN_SVR: 
	def __init__(self, deg , r=1000 ) :
		# print("RRRRR : ", r )
		self.reg = r 
		self.degree = deg
		self.training_data = []
		self.alphas = []

	def calc_K_Mat(self, input_mat, degree, c ) : 
		k_mat = np.ones([input_mat.shape[0]+1 ,input_mat.shape[0]+1 ])
		k_mat[0][0] = 0
		i_matrix = np.identity(input_mat.shape[0])
		for i in range(1, k_mat.shape[0]) : 
		    for j in range (1, k_mat.shape[0]) : 
		    	print("Fitting : K Matrix ( {} , {} )".format(i , j))
		    	k = (np.sum((input_mat[i-1:i , : ]).T * input_mat[j-1:j , :]) ** degree)  + (1/c) * i_matrix[i-1][j-1]
		    	k_mat[i][j] = k
		return k_mat

	def fit(self, input_data , target) : 
		self.training_data = np.copy(input_data)
		# print("Training data shape : " , self.training_data.shape)
		k_matrix = self.calc_K_Mat(input_data ,self.degree , self.reg)
		target = np.insert(target, 0 , 0)
		p1 = inv(np.matmul(k_matrix.T , k_matrix ))
		p2 = np.matmul( k_matrix.T , target)
		alphas = np.matmul(p1 ,p2 )
		self.alphas = alphas

	def calc_K_matrix_input(self, input_data ): 
		input_k_matrix = np.ones([input_data.shape[0] +1 , self.training_data.shape[0] + 1] )
		input_k_matrix[0][0] = 0 
		for i in range(1, input_k_matrix.shape[0]):
			for j in range(1, input_k_matrix.shape[1]):
				print("Predicting : K Matrix ( {} , {} )".format(i , j))
				input_k_matrix[i][j] = (np.sum((input_data[i-1:i , : ]).T * self.training_data[j-1:j , :] ) ** self.degree)

		return input_k_matrix

		return []
	def predict(self , input_data) : 
		# print(self.reg)
		# print(str(1e3))
		k_mat= self.calc_K_matrix_input(input_data)
		# print(k_mat.shape)
		# return np.matmul( k_mat[1: , 1:], self.alphas[1:]) + self.alphas[0]
		return np.matmul( k_mat, self.alphas)


