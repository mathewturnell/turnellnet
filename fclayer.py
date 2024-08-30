import numpy as np
from layer import Layer

from matplotlib import pyplot as plt

class FCLayer(Layer):

    def __init__(self, input_size, output_size, learningRate):

        self.learningRate = learningRate

        self.x = np.zeros((1, input_size))
        self.y = np.zeros((1, output_size))

        self.delta = np.zeros((1, output_size))
        self.delta_1 = np.zeros((1, input_size))

        self.w = np.random.rand(input_size, output_size)-0.5
        self.b = np.random.rand(1, output_size) - 0.5
        self.a = np.zeros((1, output_size))
        self.df = np.zeros((1, output_size))

        self.fig, self.axs = plt.subplots(1,6)

    def forward_propagation(self, input_data, diag):

        if(np.ndim(input_data) > 1):
            self.x = np.reshape(input_data, (1,np.shape(input_data)[0]*np.shape(input_data)[1]))

        self.a = np.matmul(self.x, self.w) + self.b

        self.y = self.a
        return self.y

    def backward_propagation(self, output_error, diag):

        self.delta = output_error
        self.delta_1 = np.matmul(self.delta, np.transpose(self.w))

        self.dw = np.matmul(np.transpose(self.x), self.delta)
        self.db = self.delta

        self.w = self.w - self.learningRate * self.dw
        self.b = self.b - self.learningRate * self.db

        return self.delta_1
    
    def printState(self):

        self.axs[0].plot(range(np.size(self.w)), np.reshape(self.w, (np.size(self.w))))
        self.axs[1].plot(range(np.size(self.dw)), np.reshape(self.dw, (np.size(self.dw))))
        self.axs[2].plot(range(np.size(self.delta)), np.reshape(self.delta, (np.size(self.delta))))
        self.axs[3].plot(range(np.size(self.delta_1)), np.reshape(self.delta_1, (np.size(self.delta_1))))
        self.axs[4].plot(range(np.size(self.x)), np.reshape(self.x, (np.size(self.x))))
        self.axs[5].plot(range(np.size(self.y)), np.reshape(self.y, (np.size(self.y))))
        

        self.axs[0].title.set_text('W')
        self.axs[1].title.set_text('dW')
        self.axs[2].title.set_text('delta')
        self.axs[3].title.set_text('delta_1')
        self.axs[4].title.set_text('x')
        self.axs[5].title.set_text('y')







