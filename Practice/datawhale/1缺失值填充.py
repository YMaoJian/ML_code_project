import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer



#删除无关项
useless_columns=['Unnamed: 0', 'source', 'custid', 'trade_no', 'bank_card_no'
, 'id_name']

#异常值处理
def iqr_outlier(x, thre = 1.5):
    x_cl = x.copy()
    q25, q75 = x.quantile(q = [0.25, 0.75])
    iqr = q75 - q25
    top = q75 + thre * iqr
    bottom = q25 - thre * iqr
    x_cl[x_cl > top] = top
    x_cl[x_cl < bottom] = bottom
    return  x_cl
	
	
#缺失值处理--drop
df = pd.read_csv('data.csv', encoding='gbk')
#查看数据缺失情况
df.isnull().sum().sort_values(ascending=False)
df.dropna(inplace=True)

label = df['status']
df.drop(columns=['status'], inplace=True)

#删除无关数据
df.drop(columns=useless_columns, inplace=True)
#reg_preference_for_trad转化成数值型
df['reg_preference_for_trad'] = df['reg_preference_for_trad'].map({'境外':0,'一线城市':1, '二线城市':2, '三线城市':3, '其他城市':4})

df_num = df.select_dtypes(include='number').copy()
df_str = df.select_dtypes(exclude='number').copy()

#对数值型数据使用箱型去除异常值
df_num_cl = pd.DataFrame()
for col in df_num.columns:
    df_num_cl[col] = iqr_outlier(df_num[col])
df_num = df_num_cl.copy()

#时间类型转化
df_date = pd.DataFrame()
df_date['latest_query_time_year'] = pd.to_datetime(df_str['latest_query_time']).dt.year
df_date['latest_query_time_month'] = pd.to_datetime(df_str['latest_query_time']).dt.month
df_date['latest_query_time_weekday'] = pd.to_datetime(df_str['latest_query_time']).dt.weekday

df_date['loans_latest_time_year'] = pd.to_datetime(df_str['loans_latest_time']).dt.year
df_date['loans_latest_time_month'] = pd.to_datetime(df_str['loans_latest_time']).dt.month
df_date['loans_latest_time_weekday'] = pd.to_datetime(df_str['loans_latest_time']).dt.weekday

df = pd.concat([df_num, df_date], axis=1, sort=False)

#模型训练
X_train, X_test, Y_train, Y_test = train_test_split(df.values, label.values, test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(
random_state =2018))])
pipe_lr.fit(X_train, Y_train)
pipe_lr.score(X_test, Y_test)
#1.0


'''
缺失值处理--中位数
'''
df = pd.read_csv('data.csv', encoding='gbk')
df.fillna(df.median(), inplace=True)
'''
使用impute填充缺失值
imputer = Imputer(strategy="median")
imputer.fit(df)
df = imputer.transform(df)
df = pd.DataFrame(df,columns=df.columns)
'''

label = df['status']
df.drop(columns=['status'], inplace=True)
#删除无关数据
df.drop(columns=useless_columns, inplace=True)
#reg_preference_for_trad转化成数值型
df['reg_preference_for_trad'] = df['reg_preference_for_trad'].map({'境外':0,'一线城市':1, '二线城市':2, '三线城市':3, '其他城市':4})

df_num = df.select_dtypes(include='number').copy()
df_str = df.select_dtypes(exclude='number').copy()

#对数值型数据使用箱型去除异常值
df_num_cl = pd.DataFrame()
for col in df_num.columns:
    df_num_cl[col] = iqr_outlier(df_num[col])
df_num = df_num_cl.copy()

#时间类型转化
df_date = pd.DataFrame()
df_date['latest_query_time_year'] = pd.to_datetime(df_str['latest_query_time']).dt.year
df_date['latest_query_time_month'] = pd.to_datetime(df_str['latest_query_time']).dt.month
df_date['latest_query_time_weekday'] = pd.to_datetime(df_str['latest_query_time']).dt.weekday

df_date['loans_latest_time_year'] = pd.to_datetime(df_str['loans_latest_time']).dt.year
df_date['loans_latest_time_month'] = pd.to_datetime(df_str['loans_latest_time']).dt.month
df_date['loans_latest_time_weekday'] = pd.to_datetime(df_str['loans_latest_time']).dt.weekday

df = pd.concat([df_num, df_date], axis=1, sort=False)

#模型训练
X_train, X_test, Y_train, Y_test = train_test_split(df.as_matrix(), label.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(
random_state =2018))])
pipe_lr.fit(X_train, Y_train)
pipe_lr.score(X_test, Y_test)
#1.0
