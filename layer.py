import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input):
        pass
    def backward(self,gradient,learning_rate):
        pass