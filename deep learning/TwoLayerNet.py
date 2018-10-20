import numpy as np
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def fit(self, x_train, y_train, lr = 1e-4, iters = 500000, reg = 1e-2):
        loss_history = []
        row, column = x_train.shape
        for i in range(iters):
            Z1 = x_train.dot(self.params['W1']) + self.params['b1']
            A1 = np.maximum(0, Z1)
            S = A1.dot(self.params['W2']) + self.params['b2']
            S = S - np.max(S, axis=1).reshape(-1, 1)
            Softmax_output = np.exp(S) / np.sum(np.exp(S), axis=1).reshape(-1, 1)
            loss = -np.sum(np.log(Softmax_output[range(row), list(y_train)]))
            loss /= row
            loss += 0.5 * reg * np.sum(self.params['W1'] * self.params['W1']) \
                    + 0.5 * reg * np.sum(self.params['W2'] * self.params['W2'])
            
            dS = Softmax_output.copy()
            dS[range(row), list(y_train)] += -1
            dS /= row
            dW2 = np.dot(A1.T, dS)
            db2 = np.sum(dS, axis=0)
            dA1 = np.dot(dS, self.params['W2'].T)
            dZ1 = dA1 * (A1 > 0)
            dW1 = np.dot(x_train.T, dZ1)
            db1 = np.sum(dZ1, axis=0)
            dW2 += reg * self.params['W2']
            dW1 += reg * self.params['W1']

            self.params['W1'] += -lr * dW1
            self.params['b1'] += -lr * db1
            self.params['W2'] += -lr * dW2
            self.params['b2'] += -lr * db2
            
            loss_history.append(loss)
        return loss_history
    
    def predict(self, x_test):
        Z1 = x_test.dot(self.params['W1']) + self.params['b1']
        A1 = np.maximum(0, Z1)    # ReLU function
        scores = A1.dot(self.params['W2']) + self.params['b2']
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
    y_train = np.array(y_train).flatten()#reshape(-1,1)
    y_test = np.array(y_test).flatten()
    
    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = titanic_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(np.unique(y_train))
nn = TwoLayerNet(7,5,2)
loss_history = nn.fit(x_train, y_train)
y_pred = nn.predict(x_test)
score = nn.score(y_test, y_pred)
print(score)
plt.plot(loss_history)
plt.ylabel("LOSS")
plt.xlabel("iters")
plt.show()
