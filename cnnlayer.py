import numpy as np
from layer import Layer
import scipy as sp
import os

from matplotlib import pyplot as plt

class CNNLayer(Layer):

    def __init__(self, input_size_x, input_size_y,  input_depth, kernel_size_x, kernel_size_y, kernel_count):

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y

        self.input_depth = input_depth
        self.kernel_count = kernel_count

        self.sizeOutput_x = input_size_x + kernel_size_x - 1
        self.sizeOutput_y = input_size_y + kernel_size_y - 1

        self.x = np.zeros((input_size_x, input_size_y, self.input_depth))
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')

        self.w = np.random.rand(kernel_size_x, kernel_size_y, input_depth, self.kernel_count)
        self.dw = np.zeros((kernel_size_x, kernel_size_y, input_depth, self.kernel_count))
        self.b = np.random.rand(self.sizeOutput_x, self.sizeOutput_y, self.kernel_count)
        self.db = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.y = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.delta = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.delta_1 = np.zeros((input_size_x, input_size_y, self.input_depth))
        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.df = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.fig, self.axs = plt.subplots(1,6)

    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')
        self.x_plus = self.x_plus[:,:, self.kernel_size_x-1:self.input_depth+self.kernel_size_x-1]
        # self.x_plus = np.copy(self.x[:,:,1])


        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        temp = np.zeros(np.shape(self.a))

        for k in range(self.kernel_count):
            self.a[:,:,k] = self.b[:,:,k]
            for d in range(self.input_depth):
                self.a[:, :, k] += sp.signal.convolve2d(np.squeeze(self.x_plus[:,:,d]), np.squeeze(self.w[:,:,d,k]), mode='valid')

        self.y = self.a

        self.printState()

        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = np.copy(output_error)
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))

        temp = np.zeros(np.shape(self.delta_1))

        for d in range(self.input_depth):
            for k in range(self.kernel_count):

                self.delta_1[:, :, d] += sp.signal.convolve2d(np.squeeze(self.delta[:,:,k]), np.squeeze(self.w[:,:,d,k]), mode='valid')

        temp = np.zeros(np.shape(self.dw))

        for d in range(self.input_depth):
            for k in range(self.kernel_count):
                self.dw[:, :, d, k] = sp.signal.convolve2d(np.squeeze(self.x_plus[:,:,d]), np.squeeze(self.delta[:,:,k]), mode='valid')
                
        self.db = self.delta

        #for k in range(self.kernel_count):
        #    self.db[:,:,k] = self.delta[:,:,k]

        self.w = self.w - learning_rate*self.dw
        self.b = self.b - learning_rate*self.db

        # self.printState()



        return self.delta_1
    
    def printState(self):

        
        d = 0
        k = 0

        self.axs[0].imshow(np.squeeze(self.w[:, :, d, k]), interpolation='nearest')
        self.axs[1].imshow(np.squeeze(self.dw[:, :, d, k]), interpolation='nearest')
        self.axs[2].imshow(np.squeeze(self.delta[:,:,k]), interpolation='nearest')
        self.axs[3].imshow(np.squeeze(self.delta_1[:,:,d]), interpolation='nearest')
        self.axs[4].imshow(np.squeeze(self.x_plus[:,:,d]), interpolation='nearest')
        self.axs[5].imshow(np.squeeze(self.y[:,:,k]), interpolation='nearest')
        

        self.axs[0].title.set_text('W[d=%d,k=%d]'%(d,k))
        self.axs[1].title.set_text('dW[d=%d,k=%d]'%(d,k))
        self.axs[2].title.set_text('delta[k=%d]'%(k))
        self.axs[3].title.set_text('delta_1[d=%d]'%(d))
        self.axs[4].title.set_text('x_plus[d=%d]'%(d))
        self.axs[5].title.set_text('y[k=%d]'%(k))

        # self.axs[0].legend()

        plt.show()
        






