#参考机器学习实战
import numpy as np

class SVM():
    def __init__(self, x_train, y_train, C, toler, kTup):
        self.x = x_train
        self.y = y_train
        self.C = C
        self.tol = toler
        self.m = np.shape(x_train)[0]
        self.alpha = np.zeros(self.m, 1)
        self.b = 0
        self.ecache = np.zeros(self.m, 2)
        self.k = np.zeros(self.m, self.m)
        for i in range(self.m):
            self.k[:, i] = self.Kfunc(self.x_train, self.x_train[i, :], kTup)
        
    def Kfunc(x, A, kTup):
        K = np.array(self.m ,1)
        if kTup[0] == 'lin': 
            K = x * A.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                delta = x[j, :] - A
                K[j] = np.dot(delta, delta.T)
            K = exp(K/(-1 * kTup[1] * 2))
        else :
            raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

    def calcEk(self, k):
        fXk = float(np.multiply(self.alpha, self.y_train).T * (self.k[:, k] + self.b
        Ek = fXk - float(self.y_train[k])
        return Ek
        
    def selectJ(self, i, Ei):
        maxK = -1;maxDeltaE = 0;Ej = 0;
        self.ecache[i] = [1, Ei]
        validEcacheList = nonzero(self.ecache[:, 0].A)[0]
        if (len(validEcacheList) > 1):
            for k in validEcacheList:
                if k == i: continue
                Ek = calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE) :
                    maxk = k; maxDeltaE = deltaE;Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, self.m)
            Ej = calcEk(j)
        return j, Ej
    
    def updateEk(self, k):
        Ek = calcEk(k)
        self.ecache = [1, Ek]
        
    def clipAlpha(al, H, L):
        if aj > H:
            aj = H
        elif aj < L:
            aj = L
        return aj
    
    def innerL(self, i):
        E1 = calcEk(i)
        if ((self.y_train[i] * Ei < -self.toler) and (self.alpha[i] < self.C) or\
            (self.y_train[i] * Ei > self.toler) and (self.alpha[i] > 0)):
            j, Ej = selectJ(i, Ei)
            alphaiold = self.alpha[i].copy()
            alphajold = self.alpha[j].copy()
            if (self.y_train[i] != self.y_train[j]):
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(C, C + self.alpha[j] - self.alpha[i])
            else : 
                L = max(0, self.alpha[j] + self.alpha[i] - C)
                H = min(C, self.alpha[j] + self.alpha[i])
            if L == H:
                continue
            eta = 2.0 * self.k[i, j] - self.k[i, i] = self.k[j, j]
            if eta >= 0:
                return 0
            self.alpha[j] -= self.y_train[j] * (Ei - Ej) / eta
            self.alpha[j] = clipAlpha(self.alpha[j], H, L)
            updateEk(j)
            if (abs(self.alpha[j] - alphajold) < 0.00001):
                return 0
            self.alpha[i] += self.y_train[j] * self.alpha[i] * \
                            (alphajold - self.alpha[j])
            updateEk(i)
            b1 = self.b - Ei - self.y_train[i] * (self.alpha[i] - alphaiold) * self.k[i,i] - \
                 self.y_train[j] * (self.alpha[j] - alphajold) * self.k[i, j]
            b2 = self.b - Ej - self.y_train[i] * (self.alpha[i] - alphaiold) * self.k[i,i] - \
                 self.y_train[j] * (self.alpha[j] - alphajold) * self.k[i,j]
            if (0 < self.alpha[i]) and (C > alpha[i]):
                self.b = b1
            elif (0 < alpah[j]) and (C > self.alpha[j]):
                self.b = b2
            else :
                self.b = (b1 + b2)/2
            return 1
        else :
            return 0
    
    def SMOP(self, maxiter):
        iters = 0
        entireSet = True
        alphaPairChange = 0
        while (iters < maxiter) and ((alphaPairChange > 0) or (entireSet)):
            alphaPairChange = 0
            if entireSet:
                for i in range(m):
                    alphaPairChange += innerL(i)
                iter += 1
            else :
                nonBoundIs = nonzero(self.ecache.A > 0) * (self.ecache.A < self.C)
                for i in nonBoundIs:
                    alphaPairChange += innerL(i)
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairChange == 0:
                entireSet = True
        
    def calcWs(self, k):
        m,n = shape(self.x_train)
        w = zeros(n, 1)
        for i in range(m):
            w += np.multiply(self.alpha[i] * self.y_train[i], self.x_train[i, :].T)
        return w
            
