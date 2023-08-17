from utils import get_data, flat_scores
from sklearn.linear_model import LogisticRegression


Xtrain, Ytrain = get_data('../data/train.json')
Xtest, Ytest = get_data('../data/test.json')



log_reg = LogisticRegression(max_iter=3000)
clm=log_reg.fit(Xtrain,Ytrain)
Ypredict = clm.predict(Xtest)

# f1 = flat_scores(Ypredict, Ytest, 'f1')
pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)

print(" f1:{0}\n Precision:{1}\n recall:{2}".format(f1, pre, recall))
