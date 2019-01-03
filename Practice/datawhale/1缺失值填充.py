import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', encoding='ISO-8859-15')
df_mean = df.fillna(df.mean())
df_median = df.fillna(df.median())
df.dropna(inplace=True)

useless_columns=['Unnamed: 0', 'source', 'custid', 'trade_no', 'bank_card_no', 'reg_preference_for_trad', 'id_name', 'latest_query_time', 'loans_latest_time']

df.drop(columns=useless_columns, inplace=True)
df_mean.drop(columns=useless_columns, inplace=True)
df_median.drop(columns=useless_columns, inplace=True)

label = df['status']
df.drop(columns=['status'], inplace=True)
 
label_mean = df_mean['status']
df_mean.drop(columns=['status'], inplace=True)

label_median = df_median['status']
df_median.drop(columns=['status'], inplace=True)


X_train, X_test, Y_train, Y_test = train_test_split(df.as_matrix(), label.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state =2018))])
pipe_lr.fit(X_train, Y_train)
pipe_lr.score(X_test, Y_test)
#0.7982646420824295

X_train_mean, X_test_mean, Y_train_mean, Y_test_mean = train_test_split(df_mean.as_matrix(), label_mean.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state =2018))])
pipe_lr.fit(X_train_mean, Y_train_mean)
pipe_lr.score(X_test_mean, Y_test_mean)
#0.7820602662929222

X_train_median, X_test_median, Y_train_median, Y_test_median = train_test_split(df_median.as_matrix(), label_median.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state =2018))])
pipe_lr.fit(X_train_median, Y_train_median)
pipe_lr.score(X_test_median, Y_test_median)
#0.7820602662929222

'''
问题1、有通过决策树进行填充缺失值，但目前不知道如何实现？
问题2、将缺失值删除相关行后效果反而比用均值和中位数填充效果好，很奇怪？
问题3、使用均值填充和中位数填充分数一模一样，是否实现出现错误？
'''
