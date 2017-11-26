import numpy as np
class JN_SVR: 
	def __init__(self, deg , r) :
		self.reg = r 
		self.degree = deg
		self.alphas = []

	def calc_K_Mat(self, input_mat, degree, c ) : 
	    k_mat = np.ones([input_mat.shape[0]+1 ,input_mat.shape[0]+1 ])
	    k_mat[0][0] = 0
	    i_matrix = np.identity(input_mat.shape[0])
	    for i in range(1, k_mat.shape[0]) : 
	        for j in range (1, k_mat.shape[0]) : 
	            #Kernal 
	            k = (np.sum((input_mat[i-1:i , : ]).T * input_mat[j-1:j , :]) ** degree)  + (1/c) * i_matrix[i-1][j-1]
	#             k = ((np.sum((input_mat[i-1:i , : ]).T * input_mat[j-1:j , :]))  ** degree)  
	            k_mat[i][j] = k
	    return k_mat

	def fit(self, input_data , target) : 
		k_matrix = self.calc_K_Mat(input_data ,self.degree , self.reg)
		target = np.insert(target, 0 , 0)
		p1 = inv(np.matmul(k_matrix.T , k_matrix ))
		p2 = np.matmul( k_matrix.T , target)
		alphas = np.matmul(p1 ,p2 )
		self.alphas = alphas
	def predict(self , input_data) : 
	    k_mat= self.calc_K_Mat(input_data , self.degree ,self.reg ) 
	    return np.matmul(self.alphas[1:].T , k_mat[1: , 1:]) + self.alphas[0]




