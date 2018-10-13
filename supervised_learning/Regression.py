import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

class RidgeRegression():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * w.T.dot(w) ** 0.5   
    
class LassoRegression():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))
    

class LinearRegression():
    def __init__(self):
        pass
    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis=1)
        #w求解:xw=y=>x.T.dot(x).dot(w)=x.T.dot(y),条件如下
        if np.linalg.det(x_train.T.dot(x_train)) == 0:
            raise ValueError("x_train.T.dot(x_train)) irreversible")
        self.w = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        
    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_pred = x_test.dot(self.w)
        return y_pred
    
#数据集获取
diabetes = datasets.load_diabetes()        #载入糖尿病数据集
diabetes_x = diabetes.data[:, np.newaxis]  #获取一个特征

#数据集划分
diabetes_x = np.array(diabetes_x).reshape(442, 10)
diabetes_x_temp = diabetes_x[:, 2:3]
print(diabetes_x_temp.shape)
diabetes_x_train = diabetes_x_temp[:-20]
diabetes_x_test = diabetes_x_temp[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

clf = LinearRegression()
clf.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_pred = clf.predict(diabetes_x_test)

plt.title(u'LinearRegression Diabetes')
plt.xlabel(u'Attributes')
plt.ylabel(u'Measure of disease')
plt.scatter(diabetes_x_test, diabetes_y_test, color = 'black')
plt.xticks(())
plt.yticks(())
plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth = 3)
plt.show()
