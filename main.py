import numpy as np 
from sklearn.svm import SVR
from sklearn.linear_model import Ridge 
import matplotlib.pyplot as plt
from random import sample
from sklearn.svm import SVR
from svr import JN_SVR
from lr import JN_Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


target_index = 59
predictor_indices = [ i for i in range(44, 59) ]

#reading the csv file
fileName = 'OnlineNewsPopularity.csv';
news_articles = np.loadtxt(fileName, dtype = float, delimiter = ',', skiprows = 1, usecols=range(1,61) )

#getting feature names 
all_features = []
with open(fileName, 'r') as f:
    all_features = f.readline().split(',')

predictors = [all_features[i] for i in predictor_indices]



# data
data = news_articles[:,predictor_indices]
target = news_articles[:, target_index]

#splitting 
# extract test data
te_indices = sample(range(len(data)), int(0.2*len(data)))
testing_data = data[te_indices]
testing_data_target = target[te_indices] 


te_data = testing_data
te_target = testing_data_target


training_data = np.delete(data, te_indices, 0)
testing_data_target = np.delete(target, te_indices, 0)

tr_data = training_data
tr_target = testing_data_target


# # 
#SVR
#USING OUR OWN IMPLEMENTATION
SVR_MODEL = JN_SVR(2 , 0.1)
SVR_MODEL.fit(tr_data , tr_target)
y_p = SVR_MODEL.predict(te_data)

#USING SCIKIT LEARN 
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_poly = svr_poly.fit(tr_data , tr_target).predict(te_data)
#MEASURING ERROR 
print("SUPPORT VECTOR REGRESSION: ON TESTING DATA\n++++++++++++++++++++++++++++++++++++++++\n")
print("MEAN SQUARED ERROR")
print("Using Scikit Learn : " , mean_squared_error(te_target , y_poly))
print("Uing OUR IMPLEMENTATION : " , mean_squared_error( te_target, y_p[1:]))

print("MEAN ABSOLUTE ERROR")
print("Using Scikit Learn : " , mean_absolute_error(te_target , y_poly))
print("Using OUR IMPLEMENTATION : " , mean_absolute_error(te_target, y_p[1:]))


#Ridge 
#using our implementation 
alphas = np.arange(0, 1.0, 0.05)
ridge_mse_list = np.zeros(len(alphas))
num_iter = 100
# alphas = [0.0001, 0.001, 0.01, 0.1, 0.5]
min_alpha = sys.maxsize
min_mse = sys.maxsize

for i in range(num_iter):
	print(i)
	# get a random 20% for validation
	v_indices = sample(range(len(data)), num_to_remove)
	v_data = data[v_indices]
	v_target = target[v_indices]
	tr_data = np.delete(data, v_indices, 0)
	tr_target = np.delete(target, v_indices, 0)

	# try all alphas
	for index, curr_alpha in enumerate(alphas):
		curr_ridge = JN_Ridge(alpha=curr_alpha)
		curr_ridge.fit(tr_data, tr_target)
		curr_ridge_predicted = curr_ridge.predict(v_data)
		curr_mse = mean_squared_error(v_target, curr_ridge_predicted)
		ridge_mse_list[index] += curr_mse

ridge_mse_list /= num_iter

min_alpha = alphas[np.argmin(ridge_mse_list)]

plt.plot(alphas, ridge_mse_list, label="MSE")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.legend()
plt.title("Ridge Regression MSE vs Training Alpha")  
plt.show()

# use alpha with lowest MSE
print("Min alpha = {}".format(min_alpha))
our_ridge = JN_Ridge(alpha=min_alpha)
our_ridge.fit(data, target)
our_ridge_predicted = our_ridge.predict(te_data)

#using scikit learn 
ridge_clf = Ridge(alpha=min_alpha).fit(data, target)
sklearn_ridge_prediction = ridge_clf.predict(te_data)

print("RIDGE REGRESSION: ON TESTING DATA\n++++++++++++++++++++++++++++++++++++++++\n")
print("MEAN SQUARED ERROR (MSE)")
print("Using Scikit Learn : " , mean_squared_error(te_target , sklearn_ridge_prediction))
print("Using OUR IMPLEMENTATION : " , mean_squared_error( te_target, our_ridge_predicted))

print("MEAN ABSOLUTE ERROR (MAE)")
print("Using Scikit Learn : " , mean_absolute_error(te_target , sklearn_ridge_prediction))
print("Using OUR IMPLEMENTATION : " , mean_absolute_error(te_target, our_ridge_predicted))

print("R-SQUARED SCORE")
print("Using Scikit Learn : " , r2_score(te_target , sklearn_ridge_prediction))
print("Using OUR IMPLEMENTATION : " , r2_score(te_target, our_ridge_predicted))



def makePlot(feature_number , test , actual , our_p , scikit_p, algorithm): 
	x = test[:,feature_number]
	print(x.shape)
	print(actual.shape)
	plt.scatter( x , te_target , color ='orange' ,label ='Actual Number of shares')
	plt.plot(x , our_p[1:] , color='blue'  , label ='OUR ' +  algorithm)
	plt.plot(x , scikit_p , color='red'  , label ='SKLEARN' +  algorithm)
	plt.xlabel(predictors[feature_number])
	plt.ylim(0, 30000)
	plt.ylabel("Number of Shares")
	plt.legend()
	plt.title( "Number of Shares  vs " + str(predictors[feature_number]))  
	plt.show()
	plt.gcf().clear()

