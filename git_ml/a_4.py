#!/usr/bin/env python3

import sys,os
import pandas as pd
import numpy as np
import time
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from imblearn.over_sampling import ADASYN



file = sys.argv[1] #"train.txt.gz"
print(file)
testfile = sys.argv[2]
print(testfile)
#file = "train.txt.gz"
data = np.loadtxt(file)
testdata = np.loadtxt(testfile)
print("data loaded")
print(len(testdata))
print(testdata.shape)
print(testdata[0:5])

#split response
X = data[:,0:351]
Y = data[:,351]
print("split response")

# split data into train and test sets
# seed = 79
# test_size = 0.333
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# print("split data")

#resample
ada = ADASYN(random_state=42, sampling_strategy=1.0, n_jobs=-1,n_neighbors=5)
X_res, y_res = ada.fit_resample(X, Y)

# fit model no training data
# model_split = XGBClassifier(njobs = -1, booster = 'gblinear', max_depth=5,
# 						 learning_rate=0.15, n_estimators=150)

model_test = XGBClassifier(n_jobs=-1, objective='binary:logistic',
							subsample=0.6, colsample_bytree=0.6,
							learning_rate=0.2,
							n_estimators=75,
							seed = 1337,
							max_depth=9,
							min_child_weight=1, eval_metric='auc')
#model_split.fit(X_train, y_train)
model_test.fit(X_res, y_res)
print("fit models")


#gridsearch
#'nthread':[4], #when use hyperthread, xgboost may become slower
#'seed': [1337]
#'n_estimators': [50] #number of trees, change it to 1000 for better results
# 'nthread':[4], #so called `eta` value
#               'max_depth': [11],
#               'min_child_weight': [10]
# parameters = {
# 			  'reg_lambda':[1.3]
#               }
# 
# 
# clf = GridSearchCV(model_test, parameters, n_jobs=-1, 
#                    cv=5, 
#                    scoring='precision',
#                    verbose=2, refit=True)
# 
# clf.fit(X_res, y_res)

#trust your CV!
# best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

#print(clf.best_params_)

# test_pred = clf.predict_proba(testdata)
test_pred = model_test.predict_proba(testdata)

# make predictions for test data
#split_pred = model_split.predict_proba(X_test)
#>>>#test_pred = model_test.predict_proba(testdata)
#split_predictions = [value[1] for value in split_pred]
test_predictions = [value[1] for value in test_pred]
print("predictions made")

# evaluate predictions
# roc = roc_auc_score(y_test, split_predictions)
# print("ROC AUC: %.2f%%" % (roc))

#output
np.savetxt(sys.argv[3],test_predictions,fmt='%g')
print("output done")
