import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LinearRegression():
    def __init__(self):
        pass
    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis=1)
        self.w = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        
    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_pred = x_test.dot(self.w)
        #泰坦尼克数据是0,1与回归求出数据会不一致，正确率为0，实际中应去掉
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_test
    def score(self, y_test, y_pred):
        result = np.equal(y_test, y_pred)
        print(result.sum())
        print(y_test.shape[0])
        return result.sum() / y_test.shape[0]

def main():
    test = False
    if test:
        x = np.array([[1,1,1],[2,2,2],[3,3,3]])
        y = np.array([[0.6],[1.2],[1.8]])
        lr = LinearRegression()
        t = lr.fit(x,y)
        print(t.w)
        x_test = np.array([[4,4,4]])
        y_test = lr.predict(x_test)
        print(y_test)
    else:
        titanic = pd.read_csv("titanic_train.csv")

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

        lr = LinearRegression()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        score = lr.score(y_test, y_pred)
        print(score)
    
if __name__ == "__main__":
    main()
