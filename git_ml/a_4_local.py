#!/usr/bin/env python3

import sys,os
import pandas as pd
import numpy as np
import time
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from imblearn.over_sampling import ADASYN

file = "train.txt.gz" #sys.argv[1] 
#testfile = sys.argv[2]
#file = "train.txt.gz"
data = np.loadtxt(file)
#testdata = np.loadtxt(testfile)

#split response
X = data[:,0:351]
Y = data[:,351]

# split data into train and test sets
seed = 79
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

ada = ADASYN(random_state=42, sampling_strategy=1.0, n_jobs=-1,n_neighbors=5)
X_res, y_res = ada.fit_resample(X_train, y_train)

# fit model on training data
model_test = XGBClassifier(n_jobs=-1, seed=1337)
#model_split.fit(X_train, y_train)
model_test.fit(X_res, y_res)
print("fit models")


#gridsearch
#'nthread':[4], #when use hyperthread, xgboost may become slower
#'seed': [1337]
parameters = {'objective':['binary:logistic'],
              'learning_rate': [0.065, 0.1, 0.15, 0.2], #so called `eta` value
              'max_depth': [3,5,9],
              'min_child_weight': [1,4,10,13],
              'subsample': [0.6],
              'colsample_bytree': [0.3,0.6],
              'n_estimators': [50], #number of trees, change it to 1000 for better results
			  'reg_lambda':[1.1,1.3]
			  }


clf = RandomizedSearchCV(model_test, parameters, n_jobs=-1, 
                   cv=3, n_iter=5,
                   scoring='precision',
                   verbose=2, refit=True)

clf.fit(X_res, y_res)

#trust your CV!
# best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

print(clf.best_params_)
print(clf.best_estimator_)

y_pred = clf.predict_proba(X_test)

#gridsearch
# some_parameters = [{'tol': (0.001,0.01)}]
# some_gscv = GridSearchCV(lr, lr_parameters, cv=4, refit=True, n_jobs = -1, scoring='roc_auc')
# some_gscv.fit(train_img, labels)
# print(some_gscv.best_params_) 
# pred = some_gscv  .predict(test_img)

# make predictions for test data
#>>>#y_pred = model.predict_proba(X_test)
predictions = [value[1] for value in y_pred]

# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
roc = roc_auc_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("ROC AUC: %.2f%%" % (roc))


#output
np.savetxt("local_out",predictions,fmt='%g')