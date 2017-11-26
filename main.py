import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from random import sample

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

data = news_articles[:,predictor_indices]
target = news_articles[:, target_index]

num_total = len(data)
num_val = num_total // 5
num_te = num_total // 5
num_tr = num_total - num_val - num_te

# extract test data
te_indices = sample(range(len(data)), num_te)
te_data = data[te_indices]
te_target = target[te_indices]
data = np.delete(data, te_indices, 0)
target = np.delete(target, te_indices, 0)

# extract validation data
v_indices = sample(range(len(data)), num_val)
v_data = data[v_indices]
v_target = target[v_indices]

# rest is for training
tr_data = np.delete(data, v_indices, 0)
tr_target = np.delete(target, v_indices, 0)

# Generate sample data
X = np.sort(5 * np.random.rand(num_total, 1), axis=0)
y = np.sin(X).ravel()

#get relevant featuers only 
#create training/validatoin/test split 

#build our own models 

#build models using scikit-learn 

#graph results + compute error 

#done