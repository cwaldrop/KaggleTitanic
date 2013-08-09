#Kaggle Getting Started Data Analysis Competition
#Predicting the survival of passengers on the Titanic
#Cabe Waldrop
#8/5/2013

from __future__ import division
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn as sk

def preprocess(X):
	
	del X['Name']
	del X['PassengerId']
	del X['Cabin']
	del X['Ticket']

	sex = X['Sex']
	age = X['Age']
	embarked = X['Embarked']
	fare = X['Fare']
		
	sex[sex == 'male'] = 1
	sex[sex == 'female'] = 0
	
	age[pd.isnull(age)] = age.mean()

	embarked[embarked == 'C'] = 1
	embarked[embarked == 'Q'] = 2
	embarked[embarked == 'S'] = 3
	embarked[pd.isnull(embarked)] = embarked.mean()

	fare[pd.isnull(fare)] = fare.mean()

	X['Sex'] = sex
	X['Embarked'] = embarked
	X['Age'] = age
	X['Fare'] = fare
	X = (np.asarray(X)).astype(float)
	
	return (X)

train_frame = pd.read_csv('train.csv', na_values = [' '])

test_train = len(train_frame) - (len(train_frame)/3)
test_test = len(train_frame)/3

y = train_frame['Survived']
y = (np.asarray(y)).astype(float)
del train_frame['Survived']
X_train = preprocess(train_frame)

clf = linear_model.LogisticRegression()
SVC = sk.svm.SVC()
linearSVC = sk.svm.LinearSVC()
clf.fit(X_train[:test_train],y[:test_train])
SVC.fit(X_train[:test_train], y[:test_train])
linearSVC.fit(X_train[:test_train], y[:test_train])

y_clf_pred = clf.predict(X_train[:-test_test])
y_svc_pred = SVC.predict(X_train[:-test_test])
y_lsv_pred = linearSVC.predict(X_train[:-test_test])

print 'Number of mislabeled points using Logistic Regression: %d' % (y[:-test_test] != y_clf_pred).sum()
print 'Number of mislabeled points using SVM : %d' % (y[:-test_test] != y_svc_pred).sum()
print 'Number of mislabeled points using LinearSVM : %d' % (y[:-test_test] != y_lsv_pred).sum()

test_frame = pd.read_csv('test.csv', na_values = [' '])

X_test = preprocess(test_frame)

y_clf_pred = clf.predict(X_test)

y_clf_pred = pd.Series(y_clf_pred)
y_clf_pred.to_csv('logit_pred.csv', sep=',')

y_svc_pred = SVC.predict(X_test)

y_svc_pred = pd.Series(y_svc_pred)
y_svc_pred.to_csv('SVC_pred.csv', sep=',')

y_lsv_pred = pd.Series(y_lsv_pred)
y_lsv_pred.to_csv('LinearSVC_pred.csv', sep=',')





