import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from sklearn.svm import SVR
from svr import JN_SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

num_total = 100
num_train = 60
num_val = 20
num_test = 20

# indices of data in CSV (minus one to account for url being stripped)
target_index = 59
predictor_indices = [ *range(1,6), 10, *range(38, 59) ]

#reading the csv file
fileName = 'OnlineNewsPopularity.csv';
news_articles = np.loadtxt(fileName, dtype = float, delimiter = ',', skiprows = 1, usecols=range(1,61) )


# data
data = news_articles[:,predictor_indices]
target = news_articles[:, target_index]

#splitting 
tr_data = data[:int(0.8*len(data))]
tr_target = target[:len(tr_data)]

te_data = data[len(tr_data):]
te_target = target[len(tr_data):]

# print("Training " , len(tr_data) , "Testing " , len(te_data))
# print(len(tr_data) + len(te_data) == len(data))
# print("target")
# print("Training " , len(tr_target) , "Testing " , len(te_target))
# print(len(tr_target) + len(te_target) == len(target))





#USING OUR OWN IMPLEMENTATION
SVR_MODEL = JN_SVR(2 , 0.001)
SVR_MODEL.fit(tr_data , tr_target)
y_p = SVR_MODEL.predict(te_data)

#USING SCIKIT LEARN 
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_poly = svr_poly.fit(tr_data,tr_target).predict(te_data)
#MEASURING ERROR 
print("SUPPORT VECTOR REGRESSION: ON TESTING DATA\n++++++++++++++++++++++++++++++++++++++++\n")

print("MEAN SQUARED ERROR")
print("Uing Scikit Learn : " , mean_squared_error(te_target , y_poly))
print("Uing OUR IMPLEMENTATION : " , mean_squared_error( te_target, y_p))

print("MEAN ABSOLUTE ERROR")
print("Uing Scikit Learn : " , mean_absolute_error(te_target , y_poly))
print("Uing OUR IMPLEMENTATION : " , mean_absolute_error(te_target, y_p))


# #pair wise plots
# fig, axes = plt.subplots(ncols=3)
# for i, yvar in enumerate([1, 2, 3]):
#     axes[i].scatter(tr_data[:,i],tr_target)

# num_total = len(data)
# num_val = num_total // 5
# num_te = num_total // 5
# num_tr = num_total - num_val - num_te

# print("", str(num_tr), str(num_val), str(num_te))

# # extract test data
# te_indices = sample(range(len(data)), num_te)
# te_data = data[te_indices]
# te_target = target[te_indices]

# data = np.delete(data, te_indices, 0)
# target = np.delete(target, te_indices, 0)

# # extract validation data
# v_indices = sample(range(len(data)), num_val)
# v_data = data[v_indices]
# v_target = target[v_indices]
# # rest is for training
# tr_data = np.delete(data, v_indices, 0)
# tr_target = np.delete(target, v_indices, 0)

# print("Data: ", str(len(tr_data)), str(len(v_data)), str(len(te_data)))
# print("Target: ", str(len(tr_target)), str(len(v_target)), str(len(te_target)))
# print("test shape : \n" ,  te_target.shape)
# print("test: \n" ,  te_target)




#done