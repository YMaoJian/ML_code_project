import numpy as np
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt

class softmax():
    def __init__(self):
        self.w = None
    
    def fit(self, x_train,y_train,lr = 0.001, iters = 50000, reg = 0.001):
        loss_history = []
        x_train = np.insert(x_train, 0, 1, axis = 1)
        row, column = x_train.shape
        nclass = np.max(y_train) + 1
        if self.w is None:
            self.w = 0.01 * np.random.randn(column, nclass)
            
        for i in range(iters):
            s = x_train.dot(self.w)
            s = s - np.max(s, axis=1).reshape(-1,1)
            softmax_output = np.exp(s) / np.sum(np.exp(s), axis=1).reshape(-1,1)
            loss = np.sum(-np.log(softmax_output[range(row), list(y_train)]))
            loss /= row
            loss += 0.5 * reg * np.sum(self.w * self.w)
            
            dS = softmax_output
            dS[range(dS.shape[0]), list(y_train)] += -1
            dw = np.dot(x_train.T, dS)
            dw /= row
            dw += reg * self.w
            
            loss_history.append(loss)
            self.w += -lr * dw

        return loss_history
        
    def predict(self,x_test):
        x_test = np.insert(x_test, 0, 1, axis = 1)
        scores = np.dot(x_test, self.w)
        y_pred = np.argmax(scores, axis=1)

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
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    
    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = titanic_data()
softmax = softmax()
loss_history = softmax.fit(x_train, y_train)
y_pred = softmax.predict(x_test)
score = softmax.score(y_test, y_pred)
print(score)
plt.plot(loss_history)
plt.ylabel("LOSS")
plt.xlabel("iters")
plt.show()
