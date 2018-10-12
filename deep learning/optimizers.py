import numpy as np
import matplotlib.pyplot as plt

class momentum():
    def __init__(self, gamma = 0.9,learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.w_update = None
    def __update__(self, w, w_grad):
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))
        self.w_update = self.gamma * self.w_update + (1 - self.gamma) * w_grad
        return w - self.learning_rate * self.w_update
        
print('------------------------------momentum--------------------------------')
J = lambda w:1.5*w**4 -15*w**3 + 3*w**2
w = 1
W = []
Loss = []
m = momentum(learning_rate=0.001, gamma = 0.1)
for i in range(100):
    w_update = (6*w**3 - 45*w**2 + 6*w) * (1+ 10 * np.random.random())
    w = m.__update__(w, w_update)
    W.append(w)
    Loss.append(J(w))
plt.plot(W)
plt.figure()
plt.plot(Loss)
plt.figure()
print(w)

#解决出现多个参数的拟合速度不一致问题  缺点：因为梯度要除以距离，长时间优化时距离越大，梯度就越来越小，效率低迷
class Ada():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.s = None
    def __update__(self, w, w_grad):
        if self.s is None:
            self.s = np.zeros(np.shape(w))
        self.s += np.power(w_grad, 2)
        w = w - self.learning_rate * (w_grad / np.sqrt(self.s))
        return w

print('------------------------------Ada--------------------------------')
J = lambda w1, w2:w1**2 + 10*w2**2
w1 = 1
w2 = -1
w = [[w1,w2]]
W = []
Loss = []
ada = Ada(learning_rate=0.1)
for i in range(200):
    w_grad = np.array([[2*w[0][0], 20*w[0][1]]])
    w = ada.__update__(w, w_grad)
    W.append(w)
    Loss.append(J(w[0][0], w[0][1]))

W = np.array(W).reshape(-1,2)
plt.plot(W[:,0])
plt.plot(W[:,1])
plt.figure()
plt.plot(Loss)
print(w)

#测试初期因为加了s_correct所以拟合速度会加快,用动量的流平均思想，到了一定时间，总的里程开始不变,解决Ada后期效率低迷问题
class RMSProp():
    def __init__(self, beta = 0.9, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.beta = beta
        self.s = None
        
    def __update__(self, w, w_grad, i):
        if self.s is None:
            self.s = np.zeros(np.shape(w))
        self.s = self.beta * self.s + (1 - self.beta) * (w_grad**2)
        s_correct = self.s/(1 - self.beta ** (i+1))
        w = w - self.learning_rate * w_grad / np.sqrt(s_correct)
        return w

print('------------------------------RMSProp--------------------------------')
J = lambda w1, w2:w1**2 + 10*w2**2
w1 = 1
w2 = -1
w = [[w1,w2]]
W = []
Loss = []
rmsp = RMSProp(learning_rate=0.01)
for i in range(200):
    w_grad = np.array([[2*w[0][0], 20*w[0][1]]])
    w = rmsp.__update__(w, w_grad, i)
    W.append(w)
    
    Loss.append(J(w[0][0], w[0][1]))

W = np.array(W).reshape(-1,2)
plt.plot(W[:,0])
plt.plot(W[:,1])
plt.figure()
plt.plot(Loss)
print(w)

#Adam 集合前面几个算法的优点
class Adam():
    def __init__(self, beta = 0.99, gamma = 0.9, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.s = None
        self.v = None
        
    def __update__(self, w, w_grad, i):
        if self.s is None:
            self.s = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        self.v = self.gamma * self.v + (1 - self.gamma) * w_grad
        v_correct = self.v / (1 - self.gamma ** (i+1))
        
        self.s = self.beta * self.s + (1 - self.beta) * (w_grad**2)
        s_correct = self.s/(1 - self.beta ** (i+1))
        
        w = w - self.learning_rate * (v_correct/np.sqrt(self.s))       #拟合速度第二
        
        #这个拟合方式相较于第二会略微的不平滑，拟合速度第一
        #s_correct = self.s/(1 + self.beta ** (i+1))
        #w = w - self.learning_rate * (v_correct/np.sqrt(s_correct))
        return w
        
print('------------------------------Adam--------------------------------')
J = lambda w:1.5*w**4 -15*w**3 + 3*w**2
w = 1
W = []
Loss = []
adam = Adam(learning_rate=0.1, gamma = 0.1)
for i in range(20):
    w_update = 6*w**3 - 45*w**2 + 6*w
    w = adam.__update__(w, w_update, i)
    W.append(w)
    Loss.append(J(w))
plt.plot(W)
plt.figure()
plt.plot(Loss)
plt.figure()
print(w)
