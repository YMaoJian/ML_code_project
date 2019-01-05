import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', encoding='ISO-8859-15')

#查看数据缺失情况
df.isnull().sum().sort_values(ascending=False)

#缺失值处理
df_mean = df.fillna(df.mean())
df_median = df.fillna(df.median())
df.dropna(inplace=True)
'''
或者使用impute
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="most_frequent")
imputer.fit(fi_copy_num)
fi_num = imputer.transform(fi_copy_num)
fi_num = pd.DataFrame(fi_num,columns=fi_copy_num.columns)
'''

label = df['status']
df.drop(columns=['status'], inplace=True)

label_mean = df_mean['status']
df_mean.drop(columns=['status'], inplace=True)

label_median = df_median['status']
df_median.drop(columns=['status'], inplace=True)


#删除无关项
useless_columns=['Unnamed: 0', 'source', 'custid', 'trade_no', 'bank_card_no'
, 'id_name']
df.drop(columns=useless_columns, inplace=True)
df_mean.drop(columns=useless_columns, inplace=True)
df_median.drop(columns=useless_columns, inplace=True)

#数据处理
df['reg_preference_for_trad'] = df['reg_preference_for_trad'].map({'境外':0,'一线城市':1, '二线城市':2, '三线城市':3})
df_mean['reg_preference_for_trad'] = df_mean['reg_preference_for_trad'].map({'境外':0,'一线城市':1, '二线城市':2, '三线城市':3})
df_median['reg_preference_for_trad'] = df_median['reg_preference_for_trad'].map({'境外':0,'一线城市':1, '二线城市':2, '三线城市':3})

df_num = df.select_dtypes(include='number').copy()
df_str = df.select_dtypes(exclude='number').copy()

df_mean_num = df_mean.select_dtypes(include='number').copy()
df_mean_str = df_mean.select_dtypes(exclude='number').copy()

df_median_num = df_median.select_dtypes(include='number').copy()
df_median_str = df_median.select_dtypes(exclude='number').copy()


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
df_num_cl = pd.DataFrame()
for col in df_num.columns:
    df_num_cl[col] = iqr_outlier(df_num[col])
df_num = df_num_cl.copy()

df_mean_num_cl = pd.DataFrame()
for col in df_mean_num.columns:
    df_mean_num_cl[col] = iqr_outlier(df_mean_num[col])
df_mean_num = df_mean_num_cl.copy()

df_median_num_cl = pd.DataFrame()
for col in df_median_num.columns:
    df_median_num_cl[col] = iqr_outlier(df_median_num[col])
df_median_num = df_median_num_cl.copy()


#时间类型转化
df_date = pd.DataFrame()
df_date['latest_query_time_year'] = pd.to_datetime(df_str['latest_query_time']).dt.year
df_date['latest_query_time_month'] = pd.to_datetime(df_str['latest_query_time']).dt.month
df_date['latest_query_time_weekday'] = pd.to_datetime(df_str['latest_query_time']).dt.weekday

df_date['loans_latest_time_year'] = pd.to_datetime(df_str['loans_latest_time']).dt.year
df_date['loans_latest_time_month'] = pd.to_datetime(df_str['loans_latest_time']).dt.month
df_date['loans_latest_time_weekday'] = pd.to_datetime(df_str['loans_latest_time']).dt.weekday


df_mean_date = pd.DataFrame()
df_mean_date['latest_query_time_year'] = pd.to_datetime(df_mean_str['latest_query_time']).dt.year
df_mean_date['latest_query_time_month'] = pd.to_datetime(df_mean_str['latest_query_time']).dt.month
df_mean_date['latest_query_time_weekday'] = pd.to_datetime(df_mean_str['latest_query_time']).dt.weekday
df_mean_date['loans_latest_time_year'] = pd.to_datetime(df_mean_str['loans_latest_time']).dt.year
df_mean_date['loans_latest_time_month'] = pd.to_datetime(df_mean_str['loans_latest_time']).dt.month
df_mean_date['loans_latest_time_weekday'] = pd.to_datetime(df_mean_str['loans_latest_time']).dt.weekday


df_median_date = pd.DataFrame()
df_median_date['latest_query_time_year'] = pd.to_datetime(df_median_str['latest_query_time']).dt.year
df_median_date['latest_query_time_month'] = pd.to_datetime(df_median_str['latest_query_time']).dt.month
df_median_date['latest_query_time_weekday'] = pd.to_datetime(df_median_str['latest_query_time']).dt.weekday
df_median_date['loans_latest_time_year'] = pd.to_datetime(df_median_str['loans_latest_time']).dt.year
df_median_date['loans_latest_time_month'] = pd.to_datetime(df_median_str['loans_latest_time']).dt.month
df_median_date['loans_latest_time_weekday'] = pd.to_datetime(df_median_str['loans_latest_time']).dt.weekday

df = pd.concat([df_num, df_date], axis=1, sort=False)
df_mean = pd.concat([df_mean_num, df_mean_date], axis=1, sort=False)
df_median = pd.concat([df_median_num, df_median_date], axis=1, sort=False)


X_train, X_test, Y_train, Y_test = train_test_split(df.as_matrix(), label.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(
random_state =2018))])
pipe_lr.fit(X_train, Y_train)
pipe_lr.score(X_test, Y_test)
#0.7982646420824295

X_train_mean, X_test_mean, Y_train_mean, Y_test_mean = train_test_split(
df_mean.as_matrix(), label_mean.as_matrix(), test_size=0.3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(
random_state =2018))])
pipe_lr.fit(X_train_mean, Y_train_mean)
pipe_lr.score(X_test_mean, Y_test_mean)
#0.7820602662929222

X_train_median, X_test_median, Y_train_median, Y_test_median = 
train_test_split(df_median.as_matrix(), label_median.as_matrix(), test_size=0.
3,random_state=2018)
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(
random_state =2018))])
pipe_lr.fit(X_train_median, Y_train_median)
pipe_lr.score(X_test_median, Y_test_median)
#0.7820602662929222
