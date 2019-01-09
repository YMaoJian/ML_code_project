import pandas as pd
from pandas import DataFrame as df
from numpy import log
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import StackingCVClassifier


base_clf = LogisticRegression(penalty='l1', C=0.1, tol=0.0001)
clfs = {
    'lr': LogisticRegression(penalty='l1', C=0.1, tol=0.0001),
    'svm': LinearSVC(C=0.05, penalty='l2', dual=True),
    'svm_linear': SVC(kernel='linear', probability=True),
    'svm_ploy': SVC(kernel='poly', probability=True),
    'bagging': BaggingClassifier(base_estimator=base_clf, n_estimators=60, max_samples=1.0, max_features=1.0,
                                 random_state=1, n_jobs=1, verbose=1),
    'rf': RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=9),
    'adaboost': AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, algorithm='SAMME'),
    'gbdt': GradientBoostingClassifier(),
    'xgb': xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=50),
    'lgb': lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, max_depth=5, n_estimators=250, num_leaves=90)
}

def show_metric(model, x_train, y_train, x_test, y_test):
  model.fit(x_train, y_train)
  y_valid = model.predict(x_test)
  
  if hasattr(model,'decision_function'):
        y_valid_proba = model.decision_function(x_test)
  else:
        y_valid_proba = model.predict_proba(x_test)[:,1]
  
  #精确度
  accurate = metrics.accuracy_score(y_test, y_valid)
  print("精确度:", accurate)
  
  #召回率
  recall = metrics.recall_score(y_test, y_valid)
  print("召回率:", recall)
  
  #F1-score
  f1score = metrics.f1_score(y_test, y_valid)
  print("F1-score:", f1score)
  
  #AUC
  auc = metrics.roc_auc_score(y_test, y_valid_proba)
  print("AUC:", auc)
 


data = pd.read_csv('2select.csv')
y_data = pd.DataFrame(data['status']).values.ravel()
x_data = data.drop(columns='status').values

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=2018)

lr_clf = clfs["lr"]
svm_clf = clfs["svm_linear"]
rf_clf = clfs["rf"]
xgb_clf = clfs["xgb"]
lgb_clf = clfs["lgb"]

pipe_lr = Pipeline([('scl',StandardScaler()),('clf',lr_clf)])
pipe_svc = Pipeline([('scl',StandardScaler()),('clf',svm_clf)])
pipe_rf = Pipeline([('scl',StandardScaler()),('clf',rf_clf)])
pipe_xgb = Pipeline([('scl',StandardScaler()),('clf',xgb_clf)])
pipe_lgb = Pipeline([('scl',StandardScaler()),('clf',lgb_clf)])

sclf = StackingCVClassifier(classifiers=[pipe_lr, pipe_svc, pipe_rf, pipe_xgb, pipe_lgb],meta_classifier=pipe_lr, use_probas=True, cv=5, verbose=3)

show_metric(sclf, x_train, y_train, x_test, y_test)

'''
精确度: 0.8047722342733189
召回率: 0.4
F1-score: 0.5054945054945055
AUC: 0.8163608946971601
'''
