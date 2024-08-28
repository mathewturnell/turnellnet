import numpy as np
from layer import Layer

class FCLayer(Layer):

    def __init__(self, input_size, output_size):

        self.x = np.zeros((1, input_size))
        self.y = np.zeros((1, output_size))

        self.delta = np.zeros((1, output_size))
        self.delta_1 = np.zeros((1, input_size))

        self.w = np.random.rand(input_size, output_size)-0.5
        self.b = np.random.rand(1, output_size) - 0.5
        self.a = np.zeros((1, output_size))
        self.df = np.zeros((1, output_size))

    def forward_propagation(self, input_data):

        if(np.ndim(input_data) > 1):
            self.x = np.reshape(input_data, (1,np.shape(input_data)[0]*np.shape(input_data)[1]))

        self.a = np.matmul(self.x, self.w) + self.b

        self.y = self.a
        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = output_error
        self.delta_1 = np.matmul(self.delta, np.transpose(self.w))

        self.w = self.w - learning_rate * np.matmul(np.transpose(self.x), self.delta)
        self.b = self.b - learning_rate * self.delta

        return self.delta_1






