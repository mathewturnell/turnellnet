#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from layer import Layer
from cnnlayer import CNNLayer
import os


from matplotlib import pyplot as plt

class Slayer(Layer):

    def __init__(self, input_size_x, input_size_y,  input_depth, factor_size_x, factor_size_y, learningRate):

        diag = 0

        self.type = "DS"

        self.learningRate = learningRate

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y

        self.kernel_size_x = 1
        self.kernel_size_y = 1

        self.factor_size_x = (int)(factor_size_x)
        self.factor_size_y = (int)(factor_size_y)

        self.input_depth = input_depth
        self.kernel_count = input_depth

        self.sizeOutput_x = (int)(self.input_size_x / factor_size_x)
        self.sizeOutput_y = (int)(self.input_size_y / factor_size_y)

        self.x = np.zeros((input_size_x, input_size_y, self.input_depth))
        self.x_plus = np.pad(self.x, self.kernel_size_x-1, mode='constant')

        self.w = 0
        self.dw = 0

        self.delta = 0


        self.delta = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.delta_1_pre = np.zeros((self.input_size_x, self.input_size_y, self.kernel_count))
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))

        self.a = np.zeros((self.input_size_x, self.input_size_y, self.kernel_count))
        self.df = np.zeros((self.input_size_x, self.input_size_x, self.kernel_count))

        self.y_pre = np.zeros((input_size_x, input_size_y, self.kernel_count))
        self.y = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        if diag == 1:
            self.fig, self.axs = plt.subplots(1,6)
            self.fig.suptitle('Layer CNN: Input %dx%d Output %dx%d\nInput depth D=%d, Output kernels K=%d '%(self.input_size_x, self.input_size_y, self.sizeOutput_x, self.sizeOutput_y,  self.input_depth, self.kernel_count), fontsize=16)

        self.numberLinesOutput = self.input_depth * self.kernel_count
        if self.numberLinesOutput > 10:
            self.numberLinesOutput = 10

    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)

        self.y_pre = self.x
        self.y = self.downSample(self.y_pre, self.y, self.factor_size_x, self.factor_size_y)

        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = np.copy(output_error)
        self.delta_1_pre = self.delta

        self.delta_1 = self.superSample(self.delta_1_pre, self.delta_1, self.factor_size_x, self.factor_size_y)

        # if diag == 1:
        #     self.printState()


        return self.delta_1
    
    def downSample(self, a, b, factor_x, factor_y):

        size_x = (int)(np.shape(a)[0])
        size_y = (int)(np.shape(a)[1])
        size_z = (int)(np.shape(a)[2])

        max = 0

        i_ref = 0
        j_ref = 0
        i_ref_1 = 0
        j_ref_1 = 0

        b = np.zeros(np.shape(b))

        for k in range(size_z):
            for i in range(size_x):
                for j in range(size_y):

                    i_ref = (int)(i/factor_x)
                    j_ref = (int)(j/factor_y)
                        
                    if i_ref >= np.shape(b)[0] or j_ref >= np.shape(b)[1]:
                        continue

                    b[i_ref,j_ref,k] += a[i,j,k]/(factor_x*factor_y)

                    i_ref_1 = i_ref
                    j_ref_1 = j_ref

        return b


    def superSample(self, a, b, factor_x, factor_y):

        size_x = (int)(np.shape(a)[0]*factor_x)
        size_y = (int)(np.shape(a)[1]*factor_y)
        size_z = (int)(np.shape(a)[2])

        for k in range(size_z):
            for i in range(size_x):
                for j in range(size_y):

                    i_ref = (int)(i/factor_x)
                    j_ref = (int)(j/factor_y)

                    if i_ref >= np.shape(a)[0] or j_ref >= np.shape(a)[1]:
                        continue

                    b[i,j,k] = a[i_ref,j_ref,k]/(factor_x*factor_y)

        return b
    
    def printState(self, axs1):

        for k in range(self.kernel_count):

            if k >= 10:
                break

            axs1[k,0].xaxis.set_tick_params(labelbottom=False)
            axs1[k,1].xaxis.set_tick_params(labelleft=False)

            axs1[k,0].imshow(np.squeeze(self.x[:,:,k]), interpolation='nearest')
            axs1[k,1].imshow(np.squeeze(self.y[:,:,k]), interpolation='nearest')

            axs1[k,0].title.set_text('x[d=%d]'%(k))
            axs1[k,1].title.set_text('y[k=%d]'%(k))



