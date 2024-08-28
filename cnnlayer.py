import numpy as np
from layer import Layer
import scipy as sp

class CNNLayer(Layer):

    def __init__(self, input_size_x, input_size_y, kernel_size_x, kernel_size_y):

        self.x = np.zeros((input_size_x, input_size_y))
        self.y = np.zeros((input_size_x - kernel_size_x + 1, input_size_y - kernel_size_y + 1))

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.output_size_x = input_size_x - kernel_size_x + 1
        self.output_size_y = input_size_y - kernel_size_y + 1

        self.w = np.random.rand(kernel_size_x, kernel_size_y)-0.5
        self.w_diag = np.random.rand(kernel_size_y, kernel_size_x) - 0.5
        self.b = np.random.rand(input_size_x - kernel_size_x + 1, input_size_y - kernel_size_y + 1) - 0.5
        self.a = np.zeros((input_size_x - kernel_size_x + 1, input_size_y - kernel_size_y + 1))

        self.delta = np.zeros((input_size_x - kernel_size_x + 1, input_size_y - kernel_size_y + 1))
        self.delta_diag = np.zeros((input_size_y - kernel_size_y + 1, input_size_x - kernel_size_x + 1))

    def forward_propagation(self, input_data):

        self.x = input_data
        self.w_diag = np.flipud(np.fliplr(self.w))
        self.a = sp.signal.convolve2d(self.x, self.w_diag, mode='same')

        self.y = self.a
        return self.y

    def backward_propagation(self, output_error, learning_rate):

        #self.delta = np.reshape(output_error,self.output_size_x, self.output_size_y)
        self.delta = output_error

        self.delta_diag = np.flipud(np.fliplr(self.delta))
        self.delta_1 = sp.signal.convolve2d(self.delta, self.w_diag, mode='same')

        self.w = self.w - learning_rate * sp.signal.convolve2d(self.x, self.delta_diag, mode='same')
        self.b = self.b - learning_rate * self.delta

        return self.delta_1






