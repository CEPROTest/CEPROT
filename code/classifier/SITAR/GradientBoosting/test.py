from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from utils import get_data, flat_scores

Xtrain, Ytrain = get_data('../data/train_part.json')
Xtest, Ytest = get_data('../data/test_part.json')

clf = GradientBoostingClassifier()
clf.fit(Xtrain, Ytrain)
print(clf.score(Xtest, Ytest))
Ypredict = clf.predict(Xtest)

acc = flat_scores(Ypredict, Ytest, 'acc')
pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')

print(" ACC:{0}\n Precision:{1}\n recall:{2}".format(acc, pre, recall))


