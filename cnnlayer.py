# import numpy as np
import cupy as np
import cupyx.scipy.signal as sp
# import cupy as sp


from layer import Layer
# import scipy as sp
import os

from matplotlib import pyplot as plt

class CNNLayer(Layer):

    def __init__(self, input_size_x, input_size_y,  input_depth, kernel_size_x, kernel_size_y, kernel_count, learningRate):

        diag = 0

        self.type = "CNN"

        self.learningRate = learningRate

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y

        self.input_depth = input_depth
        self.kernel_count = kernel_count

        self.sizeOutput_x = input_size_x + kernel_size_x - 1
        self.sizeOutput_y = input_size_y + kernel_size_y - 1

        self.x = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')

        self.w = np.random.rand(self.kernel_size_x, self.kernel_size_y, self.input_depth, self.kernel_count)-0.5
        self.dw = np.zeros((self.kernel_size_x, self.kernel_size_y, self.input_depth, self.kernel_count))
        self.b = np.random.rand(self.sizeOutput_x, self.sizeOutput_y, self.kernel_count)-0.5
        self.db = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.y = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.delta = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.df = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        if diag == 1:
            self.fig, self.axs = plt.subplots(1,6)
            self.fig.suptitle('Layer CNN: Input %dx%d Output %dx%d\nInput depth D=%d, Output kernels K=%d '%(self.input_size_x, self.input_size_y, self.sizeOutput_x, self.sizeOutput_y,  self.input_depth, self.kernel_count), fontsize=16)

        self.numberLinesOutput = self.input_depth * self.kernel_count
        if self.numberLinesOutput > 10:
            self.numberLinesOutput = 10

        self.fig1, self.axs1 = plt.subplots(self.numberLinesOutput,3)

        
        # self.fig1.suptitle('Layer CNN: Input %dx%d Output %dx%d\nInput depth D=%d, Output kernels K=%d '%(self.input_size_x, self.input_size_y, self.sizeOutput_x, self.sizeOutput_y,  self.input_depth, self.kernel_count), fontsize=16)

    def forward_propagation(self, input_data, diag):

        self.x = np.copy(input_data)
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')
        self.x_plus = self.x_plus[:,:, self.kernel_size_x-1:self.input_depth+self.kernel_size_x-1]

        self.a = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        for k in range(self.kernel_count):
            self.a[:,:,k] = self.b[:,:,k]
            for d in range(self.input_depth):
                self.a[:, :, k] += sp.correlate2d(self.x_plus[:,:,d], self.w[:,:,d,k], mode='valid')

        self.y = self.a

        # if diag == 1:
        #     self.printState()

        # self.y = (self.y-np.min(self.y))/(np.max(self.y)-np.min(self.y))

        return self.y

    def backward_propagation(self, output_error, diag):

        self.delta = np.copy(output_error)
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))

        for d in range(self.input_depth):
            for k in range(self.kernel_count):

                self.delta_1[:, :, d] += sp.convolve2d(np.squeeze(self.delta[:,:,k]), np.squeeze(self.w[:,:,d,k]), mode='valid')

        for d in range(self.input_depth):
            for k in range(self.kernel_count):
                self.dw[:, :, d, k] = sp.correlate2d(np.squeeze(self.x_plus[:,:,d]), np.squeeze(self.delta[:,:,k]), mode='valid')
                
        self.db = self.delta

        self.w = self.w + self.learningRate*self.dw
        self.b = self.b + self.learningRate*self.db

        # if diag == 1:
        #     self.printState()


        return self.delta_1
    
    def getSizeOuputs(self):
        return (self.sizeOutput_x, self.sizeOutput_y)
    
    def printStateDiag(self):

            
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

    def printState(self):

        k = 0


        # for d in range(self.input_depth):

        #     if d+k == 10:
        #         break

        #     self.axs1[d+k,0].xaxis.set_tick_params(labelbottom=False)
        #     self.axs1[d+k,1].xaxis.set_tick_params(labelleft=False)
        #     self.axs1[d+k,2].xaxis.set_tick_params(labelleft=False)
        #     self.axs1[d+k,0].yaxis.set_tick_params(labelbottom=False)
        #     self.axs1[d+k,1].yaxis.set_tick_params(labelleft=False)
        #     self.axs1[d+k,2].yaxis.set_tick_params(labelleft=False)

        #     self.axs1[d+k,0].imshow(np.squeeze(self.x_plus[:,:,d]), interpolation='nearest')
        #     self.axs1[d+k,1].imshow(np.squeeze(self.w[:, :, d, k]), interpolation='nearest')
        #     self.axs1[d+k,2].imshow(np.squeeze(self.y[:,:,k]), interpolation='nearest')

        #     self.axs1[d+k,0].title.set_text('x_plus[d=%d]'%(d))
        #     self.axs1[d+k,1].title.set_text('W[d=%d,k=%d]'%(d,k))
        #     self.axs1[d+k,2].title.set_text('y[k=%d]'%(k))

        #     k = k + 1
        #     if k >= self.kernel_count:
        #         k = 0
        

        # self.axs[0].legend()

        






