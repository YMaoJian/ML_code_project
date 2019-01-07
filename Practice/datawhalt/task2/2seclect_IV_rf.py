import pandas as pd
from pandas import DataFrame as df
from numpy import log
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def cal_WOE(dataset, col, target):
    # 对特征进行统计分组
    subdata = df(dataset.groupby(col)[col].count())
    # 每个分组中响应客户的数量
    suby = df(dataset.groupby(col)[target].sum())
    # subdata 与 suby 的拼接
    data = df(pd.merge(subdata, suby, how='left', left_index=True, right_index=True))

    # 相关统计，总共的样本数量total，响应客户总数b_total，未响应客户数量g_total
    b_total = data[target].sum()
    total = data[col].sum()
    g_total = total - b_total

    # WOE公式
    data["bad"] = data.apply(lambda x:round(x[target]/b_total, 100), axis=1)
    data["good"] = data.apply(lambda x:round((x[col] - x[target])/g_total, 100), axis=1)
    data["WOE"] = data.apply(lambda x:np.log(x.bad / x.good), axis=1)
    return data.loc[:, ["bad", "good", "WOE"]]


def cal_IV(dataset):
    dataset["IV"] = dataset.apply(lambda x:(x["bad"] - x["good"]) * x["WOE"], axis=1)
    IV = sum(dataset["IV"])
    return IV

data_IV = df()
fea_iv = []

data = pd.read_csv('1process.csv')
col_list = [col for col in data.drop(labels='status', axis=1)]


for col in col_list:
        col_WOE = cal_WOE(data, col, 'status')
        col_IV = cal_IV(col_WOE)
        if col_IV > 0.1:
            data_IV[col] = [col_IV]
            fea_iv.append(col)



data_IV.to_csv('data_IV.csv', index=0)
print('IV select',fea_iv)


X = data.drop(labels='status', axis=1)
y = data['status']

rfc = RandomForestClassifier()
rfc.fit(X, y)
rfc_impc = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
fea_gini = rfc_impc[:20].index.tolist()
print('rf select', fea_gini)


features = list(set(fea_gini)|set(fea_iv))
X_final = X[features]

data_fi = pd.concat([X_final, y], axis=1)
data_fi.to_csv('2select.csv',index = False)


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


#LR
print('lr')
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2', class_weight='balanced', max_iter=100, random_state =2018))])
show_metric(pipe_lr, x_train, y_train, x_test, y_test)
'''
精确度: 0.7418655097613883
召回率: 0.7043478260869566
F1-score: 0.5765124555160144
AUC: 0.8106810756471476
'''

#SVM
print('svm')
svc = SVC(class_weight='balanced', probability=True, random_state=2018)
pipe_svc = Pipeline([('scl',StandardScaler()),('clf',svc)])
show_metric(pipe_svc, x_train, y_train, x_test, y_test)
'''
精确度: 0.754880694143167
召回率: 0.6608695652173913
F1-score: 0.5735849056603773
AUC: 0.7911032922844936
'''

#Tree
print('tree')
dtree = DecisionTreeClassifier(max_depth=30, min_samples_split=3, max_features='log2', random_state=2018, max_leaf_nodes=8, class_weight='balanced')

pipe_dtree = Pipeline([('scl',StandardScaler()),('clf',dtree)])
show_metric(pipe_dtree, x_train, y_train, x_test, y_test)
'''
精确度: 0.6377440347071583
召回率: 0.5043478260869565
F1-score: 0.4098939929328622
AUC: 0.6163860266398593
'''

#RF
print('rf')
rf = RandomForestClassifier(n_estimators=190, max_features=0.3, max_depth=10, min_samples_split=2,  min_samples_leaf=2, class_weight='balanced', max_leaf_nodes=70, random_state=2018)

pipe_rf = Pipeline([('scl',StandardScaler()),('clf',rf)])
show_metric(pipe_rf, x_train, y_train, x_test, y_test)
'''
精确度: 0.789587852494577
召回率: 0.41739130434782606
F1-score: 0.4974093264248704
AUC: 0.7753204322694144
''

#GDBT
print('gdbt')
gdbt = GradientBoostingClassifier(loss='exponential', n_estimators=100, max_depth=3,  max_features=0.2, min_samples_leaf=4, random_state=2018)
pipe_gdbt = Pipeline([('scl',StandardScaler()),('clf',gdbt)])
show_metric(pipe_gdbt, x_train, y_train, x_test, y_test)
'''
精确度: 0.7939262472885033
召回率: 0.391304347826087
F1-score: 0.4864864864864865
AUC: 0.7729831615983915
'''


#xgb
print('xgb')
xgbClassify = xgb.XGBClassifier(max_depth=10, n_estimators=20, silent=1, booster='gbtree', 
                                     gamma=0.5, subsample=0.5,random_state=2018)
pipe_xgb = Pipeline([('scl',StandardScaler()),('clf',xgbClassify)])
show_metric(pipe_xgb, x_train, y_train, x_test, y_test)
'''
精确度: 0.7917570498915402
召回率: 0.3739130434782609
F1-score: 0.47252747252747257
AUC: 0.7426489067604927
'''

#lgb
print('lgb')
lgbClassify = lgb.LGBMClassifier()
pipe_lgb = Pipeline([('scl',StandardScaler()),('clf',lgbClassify)])
show_metric(pipe_lgb, x_train, y_train, x_test, y_test)
'''
精确度: 0.7960954446854663
召回率: 0.391304347826087
F1-score: 0.4891304347826087
AUC: 0.770696154812767
'''

