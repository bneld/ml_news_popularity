
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt


#reading the csv file
fileName = 'OnlineNewsPopularity.csv';
news_articles = np.loadtxt(fileName , dtype = float , delimiter = ',' ,  skiprows = 1 )



#get relevant featuers only 
#create training/validatoin/test split 

#build our own models 

#build models using sickit learn 

#graph results + compute error 

#done