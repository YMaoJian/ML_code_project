import numpy as np
from sklearn.model_selection import train_test_split
import pandas

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

class LogisticRegression():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
###################################################################################
#gradient_function='BGD'时,num无效；gradient_function='SGD'时,num表示随机选取次数；
#gradient_function='MBGD'时，num表示选取样本的num/10次数
##################################################################################
    def fit(self, x_train, y_train, gradient_function, num):
        self.w = np.random.uniform(-0.01,0.01,size=x_train.shape[1])
        self.bias =  np.zeros((1,1))
        
        if gradient_function == 'BGD':
            for i in range(x_train.shape[0]):
                f = sigmoid(np.sum(np.multiply(x_train[i], self.w) + self.bias))
                loss = y_train[i] - f
                self.w = self.w + self.learning_rate * loss * x_train[i]
                self.bias = self.bias + self.learning_rate * loss
        elif gradient_function == 'SGD': #随机梯度下降
            for i in range(num):
                learning_rate = self.learning_rate + 0.04/(1.0+i)  #详见机器学习实战程序清单5-4
                random_index = int(np.random.uniform(0, x_train[0]))
                f = sigmoid(np.dot(x_train[random_index], self.w))
                loss = y_train[random_index] - f
                self.w = self.w + self.learning_rate * loss * x_train[random_index]
                self.bias= self.bias + self.learning_rate * loss
        elif gradient_function == 'MBGD':
            for i in range(x_train.shape[0] * num / 10):
                f = sigmoid(np.dot(x_train[i], self.w))
                loss = y_train[i] - f
                self.w = self.w + self.learning_rate * loss * x_train[i]
                self.bias = self.bias + self.learning_rate * loss
        
    def predict(self, x_test):
        y_pred = sigmoid(np.sum(np.multiply(x_test, self.w) + self.bias, axis=1))
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
    def score(self, y_test, y_pred):
        result = np.equal(y_test, y_pred)
        return result.sum() / y_test.shape[0]
    
def main():
    #这里使用泰坦尼克号数据测试，结果为0.7014925373134329，而scikit库为0.78
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
    lr = LogisticRegression(0.01)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    lr.fit(x_train, y_train, 'BGD', 1)
    y_pred = lr.predict(x_test)

    score = lr.score(y_test, y_pred)
    print(score)
if __name__ == "__main__":
    main()
