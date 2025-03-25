import numpy as np
from layer import Layer
class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        #print(self.weights)
        self.bias = np.random.randn(output_size,1)
        self.prv_dw= 0
    def forward(self,input):
        self.input  =input
        out = np.dot(self.weights,self.input) + self.bias
        return out
    def backward(self,output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
        