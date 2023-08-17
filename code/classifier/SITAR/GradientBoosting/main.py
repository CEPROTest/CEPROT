from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from utils import get_data, flat_scores

Xtrain, Ytrain = get_data('../data/train.json')
Xtest, Ytest = get_data('../data/test.json')

clf = GradientBoostingClassifier()
clf.fit(Xtrain, Ytrain)
print(clf.score(Xtest, Ytest))
Ypredict = clf.predict(Xtest)

# f1 = flat_scores(Ypredict, Ytest, 'f1')
pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)


print(" f1:{0}\n Precision:{1}\n recall:{2}".format(f1, pre, recall))


