import numpy as np
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt

class LinearSVM():
    def __init__(self):
        self.w = None
        
    def fit(self, x_train, y_train, theta=1, lamda=0.0001, lr = 0.0001, iters = 100000):
        loss_history = []
        x_train = np.insert(x_train, 0, 1, axis = 1)
        row, column = x_train.shape
        nclass = np.max(y_train) + 1
        if self.w is None:
            self.w = 0.01 * np.random.randn(column, nclass)
          
        for i in range(iters):
            s = x_train.dot(self.w)
            s_corrt = s[range(row),list(y_train)].reshape(-1, 1)
            res = s - s_corrt + theta
            margin = np.maximum(0, res)
            margin[range(row), list(y_train)] = 0
            loss = np.sum(margin) / row + lamda * np.sum(self.w * self.w)

            mask = np.zeros((row, nclass))
            mask[margin > 0] = 1
            mask[range(row), list(y_train)] = 0
            mask[range(row), list(y_train)] = -np.sum(mask, axis=1)
            dw = np.dot(x_train.T, mask)
            dw = dw / row + lamda * self.w

            loss_history.append(loss)
            self.w += -lr * dw

        return loss_history
    
    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis = 1)
        scores = x_test.dot(self.w)
        y_pred = np.argmax(scores, axis = 1)
            
        return y_pred
    
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
    y_train = np.array(y_train).flatten()#reshape(-1,1)
    y_test = np.array(y_test).flatten()
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = titanic_data()
lsvm = LinearSVM()
loss_history = lsvm.fit(x_train, y_train)
y_pred = lsvm.predict(x_test)
score = lsvm.score(y_test, y_pred)
print(score)
plt.plot(loss_history)
plt.ylabel("LOSS")
plt.xlabel("iters")
plt.show()
