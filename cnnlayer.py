#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from layer import Layer
import scipy as sp
import os

from matplotlib import pyplot as plt

class CNNLayer(Layer):

    def __init__(self, input_size_x, input_size_y,  input_depth, kernel_size_x, kernel_size_y, kernel_count, stride, learningRate):

        diag = 0

        self.type = "CNN"

        self.learningRate = learningRate

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y

        self.stride = stride

        self.input_depth = input_depth
        self.kernel_count = kernel_count

        self.sizeOutput_x = input_size_x + kernel_size_x - 1
        self.sizeOutput_y = input_size_y + kernel_size_y - 1

        self.x = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')

        self.initFactor = np.sqrt(6/(self.input_size_x**2+self.sizeOutput_x**2))

        self.w = np.random.normal(0,self.initFactor, size=(self.kernel_size_x, self.kernel_size_y, self.input_depth, self.kernel_count))
        self.dw = np.zeros((self.kernel_size_x, self.kernel_size_y, self.input_depth, self.kernel_count))
        self.b = np.random.normal(0,self.initFactor, size=(self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.db = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        
        self.y = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.delta = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.df = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.numberLinesOutput = self.input_depth * self.kernel_count
        if self.numberLinesOutput > 10:
            self.numberLinesOutput = 10

        self.limitGraphsX = 10
        self.limitGraphsY = 10



    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')
        self.x_plus = self.x_plus[:,:, self.kernel_size_x-1:self.input_depth+self.kernel_size_x-1]

        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        for k in range(self.kernel_count):
            self.a[:,:,k] = self.b[:,:,k]
            for d in range(self.input_depth):
                self.a[:, :, k] += sp.signal.correlate2d(self.x_plus[:,:,d], self.w[:,:,d,k], mode='valid')

        self.y = self.a

        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = np.copy(output_error)
        self.delta = self.delta.clip(-0.99,0.99)
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        

        for d in range(self.input_depth):
            for k in range(self.kernel_count):

                self.delta_1[:, :, d] += sp.signal.convolve2d(np.squeeze(self.delta[:,:,k]), np.squeeze(self.w[:,:,d,k]), mode='valid')

        for d in range(self.input_depth):
            for k in range(self.kernel_count):
                self.dw[:, :, d, k] = sp.signal.correlate2d(np.squeeze(self.x_plus[:,:,d]), np.squeeze(self.delta[:,:,k]), mode='valid')
                
        self.delta_1 = self.delta_1.clip(-0.99,0.99)
        self.db = self.delta

        lr = learning_rate

        self.w = self.w - lr*self.dw
        self.b = self.b - lr*self.db

        # if diag == 1:
        #     self.printState()

        return self.delta_1
    
    def getSizeOuputs(self):
        return (self.sizeOutput_x, self.sizeOutput_y)
    
    def printStateDiag(self):         
        return


    def printState(self, axs1):

        d = 0

        for k in range(self.kernel_count):

            if d+k >= 10:
                break

            axs1[d+k,0].xaxis.set_tick_params(labelbottom=False)
            axs1[d+k,1].xaxis.set_tick_params(labelleft=False)
            axs1[d+k,2].xaxis.set_tick_params(labelleft=False)
            axs1[d+k,0].yaxis.set_tick_params(labelbottom=False)
            axs1[d+k,1].yaxis.set_tick_params(labelleft=False)
            axs1[d+k,2].yaxis.set_tick_params(labelleft=False)

            axs1[d+k,0].imshow(np.squeeze(self.x_plus[:,:,d]), interpolation='nearest')
            axs1[d+k,1].imshow(np.squeeze(self.w[:, :, d, k]), interpolation='nearest')
            axs1[d+k,2].imshow(np.squeeze(self.y[:,:,k]), interpolation='nearest')

            axs1[d+k,0].title.set_text('Input: x[d=%d]'%(d))
            axs1[d+k,1].title.set_text('Kernel: W[d=%d,k=%d]'%(d,k))
            axs1[d+k,2].title.set_text('Output: y[k=%d]'%(k))

            k = k + 1
            if k >= self.kernel_count:
                k = 0

        d = 0

        






