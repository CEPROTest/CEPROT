from utils import get_data, flat_scores
import sklearn
from sklearn import naive_bayes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy

Xtrain, Ytrain = get_data('../data/train.json')
Xtest, Ytest = get_data('../data/test.json')


model =  naive_bayes.GaussianNB()  
model.fit(Xtrain,Ytrain)
Ypredict=model.predict(Xtest)

# f1 = flat_scores(Ypredict, Ytest, 'f1')
pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)

print(" f1:{0}\n Precision:{1}\n recall:{2}".format(f1, pre, recall))
