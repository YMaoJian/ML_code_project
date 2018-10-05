import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate, gradient_function):
        self.learning_rate = learning_rate
        self.gradient_function = gradient_function
