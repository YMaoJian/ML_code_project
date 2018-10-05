import numpy as np
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, x_train, y_train, k, norm):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.norm = norm
    
    def predict(self, x_test):
        num_test = x_test.shape[0]
        y_pred = np.zeros(num_test, dtype = self.y_train.dtype)

        for i in range(num_test):
            if self.norm == 'L1':
                distances = np.sum(np.abs(self.x_train - x_test[i, :]), axis = 1)
            elif self.norm == 'L2':
                distances = np.sqrt(np.sum(np.square(self.x_train - x_test[i, :]), axis = 1))  
            kdists_sort_index = np.argsort(distances)[0:self.k]
            k_label = self.y_train[kdists_sort_index]
            
            k_label_unique, k_label_count = np.unique(k_label, return_counts = True)
            y_pred[i] = k_label_unique[np.argmax(k_label_count)]
        return y_pred
