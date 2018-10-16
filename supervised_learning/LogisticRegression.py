#1、特征具有相近的尺度，这将帮助梯度下降算法更快地收敛。尽量缩放到[-1,1]，因为x0=1
#2、学习率过小，达到收敛的迭代次数会非常高；学习率过大，每次迭代可能不会减小代价函数，
#   可能会越过局部最小值导致无法收敛

import numpy as np
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def SGD(x_train, y_train, iters, learning_rate):
    w = np.random.uniform(0,0.01,size=(x_train.shape[1],1))
    weights = []
    
    for i in range(iters):
        for j in range(x_train.shape[0]):
            learning_rate = learning_rate + 0.04/(1.0+i+j)     #详见机器学习实战程序清单5-4
            random_index = int(np.random.uniform(0, x_train.shape[0]))
            
            f = sigmoid(x_train[random_index, :].reshape(1, -1).dot(w))
            loss = y_train[random_index] - f

            dw = -x_train[random_index, :].reshape(1, -1).T.dot(loss)
            w += - learning_rate * dw * (1 / x_train.shape[0])
            weights.append(w.ravel().T)
    weights = np.array(weights).reshape(-1,x_train.shape[1])

    return w, weights

def BGD(x_train, y_train, iters, learning_rate):
    w = np.random.uniform(0,0.01,size=(x_train.shape[1],1))
    weights = np.zeros((iters, x_train.shape[1]))
    
    for i in range(iters):
        f = sigmoid(np.dot(x_train, w))
        loss = y_train - f
        dw = -x_train.T.dot(loss)
        w += -learning_rate * dw * (1 / x_train.shape[0])
        weights[i,:] = w.ravel().T
        
    return w, weights

class LogisticRegression():
    def __init__(self, iters = 1000, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.iters = iters

    def fit(self, x_train, y_train, gradient_function):
        x_train = np.insert(x_train, 0, 1, axis=1)
        
        if gradient_function == 'BGD':
            self.w, self.weights = BGD(x_train, y_train, self.iters, self.learning_rate)
            
        elif gradient_function == 'SGD':
            self.w, self.weights = SGD(x_train, y_train, self.iters, self.learning_rate)
        
    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_pred = sigmoid(np.dot(x_test, self.w))
        
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
    
    def w_process(self):
        return self.weights
    
    def score(self, y_test, y_pred):
        result = np.equal(y_test, y_pred)
        return result.sum() / y_test.shape[0]
    
def titanic_data():
    titanic = pandas.read_csv("titanic_train.csv")

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"]=="male","Sex"] = 0
    titanic.loc[titanic["Sex"]=="female","Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna('S')
    titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0
    titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1
    titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2
    predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    x = titanic[predictors]
    y = titanic["Survived"]
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)
    
    return x_train, x_test, y_train, y_test

#-----------------------------测试--------------------------------------------------
x_train, x_test, y_train, y_test = titanic_data()
lr = LogisticRegression()
lr.fit(x_train, y_train, gradient_function = 'SGD')#随机梯度下降的图因为范围原因不明显
y_pred = lr.predict(x_test)
score = lr.score(y_test, y_pred)
print(score)
weights = lr.w_process()
plt.plot(weights)
plt.ylabel("w0-2")
plt.xlabel("iters")
plt.show()
