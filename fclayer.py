#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from layer import Layer
from matplotlib import pyplot as plt

class FCLayer(Layer):

    def __init__(self, input_size, output_size, learningRate):

        self.type = "FC"

        self.learningRate = learningRate

        self.x = np.zeros((1, input_size))
        self.y = np.zeros((1, output_size))

        self.input_size_x = input_size
        self.input_size_y = 1
        self.input_depth = 1
        self.kernel_count = 1
        self.sizeOutput_x = 1
        self.sizeOutput_y = output_size

        self.delta = np.zeros((1, output_size))
        self.delta_1 = np.zeros((1, input_size))

        self.initFactor = np.sqrt(6/(self.input_size_x+self.sizeOutput_y))

        self.w = np.random.normal(0, self.initFactor, (input_size, output_size))
        self.b = np.random.normal(0, self.initFactor, (1,output_size))
        self.a = np.zeros((1, output_size))
        self.df = np.zeros((1, output_size))


    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)

        if(np.ndim(input_data) > 1):
            self.x = np.reshape(input_data, (1,np.shape(input_data)[0]*np.shape(input_data)[1]))

        self.a = np.matmul(self.x, self.w) + self.b

        self.y = self.a
        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = output_error
        self.delta = self.delta.clip(-0.99,0.99)
        self.delta_1 = np.matmul(self.delta, np.transpose(self.w))
        self.delta_1 = self.delta_1.clip(-0.99,0.99)

        self.dw = np.matmul(np.transpose(self.x), self.delta)
        self.db = self.delta

        lr = learning_rate

        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db

        return self.delta_1
    

            
    
    def printState(self, axs1):

        for k in range(self.kernel_count):

            if k >= 10:
                break

            axs1[k,0].xaxis.set_tick_params(labelbottom=False)
            axs1[k,1].xaxis.set_tick_params(labelleft=False)


            axs1[k,0].plot(np.squeeze(self.x))
            axs1[k,1].plot(np.squeeze(self.y))


            axs1[k,0].title.set_text('x[d=%d]'%(k))
            axs1[k,1].title.set_text('y[k=%d]'%(k))

        return








