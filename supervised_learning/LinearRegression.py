import numpy as np

class LinearRegression():
    def __init__(self):
        pass
    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis=1)
        #w求解方式：xw=y=>x.T.dot(x).dot(w)=x.T.dot(y)与西瓜书不一致。
        self.w = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        
    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_pred = x_test.dot(self.w)
        return y_test
        
    def score(self, y_test, y_pred):
        result = np.equal(y_test, y_pred)
        return result.sum() / y_test.shape[0]
