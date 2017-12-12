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
predictor_indices = [ *range(44, 59) ]

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


te_data = testing_data[:40]
te_target = testing_data_target[:40]


training_data = np.delete(data, te_indices, 0)
testing_data_target = np.delete(target, te_indices, 0)

tr_data = training_data[:200]
tr_target = testing_data_target[:200]


# 
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
print("Uing Scikit Learn : " , mean_squared_error(te_target , y_poly))
print("Uing OUR IMPLEMENTATION : " , mean_squared_error( te_target, y_p[1:]))

print("MEAN ABSOLUTE ERROR")
print("Uing Scikit Learn : " , mean_absolute_error(te_target , y_poly))
print("Uing OUR IMPLEMENTATION : " , mean_absolute_error(te_target, y_p[1:]))


# #Ridge 
# #using our implementation 
# alpha = 0.01
# our_ridge = JN_Ridge(alpha=0.001)
# our_ridge.fit(tr_data, tr_target)
# our_ridge_predicted = our_ridge.predict(te_data)

# #using scikit learn 
# ridge_clf = Ridge(alpha = 0.001).fit(tr_data , tr_target)
# sklearn_ridge_prediction = ridge_clf.predict(te_data)

# print("RIDGE REGRESSION: ON TESTING DATA\n++++++++++++++++++++++++++++++++++++++++\n")
# print("MEAN SQUARED ERROR (MSE)")
# print("Uing Scikit Learn : " , mean_squared_error(te_target , sklearn_ridge_prediction))
# print("Uing OUR IMPLEMENTATION : " , mean_squared_error( te_target, our_ridge_predicted))

# print("MEAN ABSOLUTE ERROR (MAE)")
# print("Uing Scikit Learn : " , mean_absolute_error(te_target , sklearn_ridge_prediction))
# print("Uing OUR IMPLEMENTATION : " , mean_absolute_error(te_target, our_ridge_predicted))





# #LASSO REGRESSION
# #our implementation
# our_lasso = JN_Lasso(alpha=0.01)
# our_lasso.fit(tr_data, tr_target)
# our_lasso_predicted = our_lasso.predict(te_data)

# #sklearn implementation
# sklearn_lasso= Lasso(alpha=0.01, copy_X=True, normalize=True, max_iter=1000).fit(tr_data, tr_target)
# sklearn_lasso_predicted = sklearn_lasso.predict(te_data)


# print("LASSO REGRESSION: ON TESTING DATA\n++++++++++++++++++++++++++++++++++++++++\n")
# print("MEAN SQUARED ERROR (MSE)")
# print("Uing Scikit Learn : " , mean_squared_error(te_target , sklearn_lasso_predicted))
# print("Uing OUR IMPLEMENTATION : " , mean_squared_error( te_target, our_lasso_predicted))

# print("MEAN ABSOLUTE ERROR (MAE)")
# print("Uing Scikit Learn : " , mean_absolute_error(te_target , sklearn_lasso_predicted))
# print("Uing OUR IMPLEMENTATION : " , mean_absolute_error(te_target, our_lasso_predicted))


def makePlot(feature_number , actual , our_p , scikit_p, algorithm): 
	x = te_data[:,feature_number]
	print(x.shape)
	print(actual.shape)
	plt.scatter( x , te_target , color ='orange' ,label ='Actual Number of shares')
	plt.plot(x , our_p[1:] , color='blue'  , label ='OUR ' +  algorithm)
	plt.plot(x , scikit_p , color='red'  , label ='SKLEARN' +  algorithm)
	plt.xlabel(predictors[feature_number])
	plt.ylabel("Number of Shares")
	plt.legend()
	plt.title( "Number of Shares  vs " + str(predictors[feature_number]))  
	plt.show()
	plt.gcf().clear()

