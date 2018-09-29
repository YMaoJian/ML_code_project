import numpy as np

def dot(A, B):
    res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B[0])):
            for j in range(len(B)):
                res[i][k] += A[i][j] * B[j][k]
         
    return res

def multiply(A, B):
    res = [[0] * len(A[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            res[i][j] = A[i][j] * B[i][j]
            
    return res
