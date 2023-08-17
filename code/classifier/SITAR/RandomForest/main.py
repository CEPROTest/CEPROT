from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import get_data, flat_scores
import time 

time1 = time.time()
Xtrain, Ytrain = get_data('../data/train.json')
Xtest, Ytest = get_data('../data/test.json')

rfc = RandomForestClassifier()
rfc = rfc.fit(Xtrain,Ytrain)
time2 = time.time()

Ypredict = rfc.predict(Xtest)
pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)

print(" f1:{0}\n Precision:{1}\n recall:{2}".format(f1, pre, recall))
time3 = time.time()
print(time2 - time1)
print(time3 - time2)


