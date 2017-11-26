import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from random import sample
from sklearn.svm import SVR
from svr import JN_SVR

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


# Generate sample data
data = news_articles[:,predictor_indices]
target = news_articles[:, target_index]

num_total = len(data)
num_val = num_total // 5
num_te = num_total // 5
num_tr = num_total - num_val - num_te

print("", str(num_tr), str(num_val), str(num_te))

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

print("Data: ", str(len(tr_data)), str(len(v_data)), str(len(te_data)))
print("Target: ", str(len(tr_target)), str(len(v_target)), str(len(te_target)))
print("test shape : \n" ,  te_target[:40].shape)
print("test: \n" ,  te_target[:40])


#USING OUR OWN IMPLEMENTATION
SVR_MODEL = JN_SVR(2 , 1e3)
SVR_MODEL.fit(tr_data[:40] , tr_target[:40])
y_p = SVR_MODEL.predict(te_data[:40])

#show time
lw = 2
plt.scatter([i for i in range(len(te_target[:40]))],te_target[:40] , color='darkorange', label='data')
plt.plot([i for i in range(len(y_p))], y_p, color='red', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()



## USING SICKIT LEARN


svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_poly = svr_poly.fit(tr_data[:40] , tr_target[:40]).predict(te_data[:40])

lw = 2
plt.scatter([i for i in range(len(te_target[:40]))], te_target[:40], color='darkorange', label='data')
plt.plot( [i for i in range(len(y_poly))], y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SICKIT LEARN Support Vector Regression')
plt.legend()
plt.show()





#done