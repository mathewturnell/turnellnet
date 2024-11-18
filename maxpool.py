#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from layer import Layer
from cnnlayer import CNNLayer
import os


from matplotlib import pyplot as plt

class Maxpool(Layer):

    def __init__(self, input_size_x, input_size_y,  input_depth, stride_x, stride_y):

        diag = 0

        self.type = "Maxpool"

        self.w = 0
        self.dw = 0

        self.padded = False

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y

        self.stride_x = stride_x
        self.stride_y = stride_y

        self.input_depth = input_depth
        self.kernel_count = input_depth

        i_x = self.input_size_x / stride_x
        i_y = self.input_size_y / stride_y

        if i_x - np.floor(i_x) > 0 or i_y - np.floor(i_y) > 0:
            self.padded = True
            i_x = (int)(np.floor(i_x))
            i_y = (int)(np.floor(i_y))

        self.sizeOutput_x = (int)(i_x)
        self.sizeOutput_y = (int)(i_y)

        self.x = np.zeros((input_size_x, input_size_y, self.input_depth))
        self.mask = np.zeros((input_size_x, input_size_y, self.input_depth))

        self.delta = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))
        self.delta_1 = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))

        self.y = np.zeros((self.sizeOutput_x, self.sizeOutput_y, self.kernel_count))

        self.numberLinesOutput = self.input_depth * self.kernel_count
        if self.numberLinesOutput > 10:
            self.numberLinesOutput = 10

    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)
        self.mask = np.zeros((self.input_size_x, self.input_size_y, self.input_depth))
        self.y = self.downSample(self.x, self.stride_x, self.stride_y)

        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = np.copy(output_error)
        self.delta_1 = self.superSample(self.delta, self.stride_x, self.stride_y)

        return self.delta_1
    
    def downSample(self, a, stride_x, stride_y):

        size_x = (int)(np.floor(np.shape(a)[0]/stride_x))
        size_y = (int)(np.floor(np.shape(a)[1]/stride_y))
        size_z = (int)(np.shape(a)[2])

        b = np.zeros((size_x, size_y, size_z))

        for k_b in range(size_z):
            for i_b in range(size_x):
                    for j_b in range(size_y):

                        i_ref = -1
                        j_ref = -1
                        k_ref = k_b
                        k_a = k_b

                        for i_s in range(stride_x):
                            for j_s in range(stride_y):

                                i_a = i_b * stride_x + i_s
                                j_a = j_b * stride_y + j_s                    

                                if i_a >= np.shape(a)[0] or j_a >= np.shape(a)[1]:
                                    continue

                                if i_ref == -1 or j_ref == -1:
                                    i_ref = i_a
                                    j_ref = j_a
                                    b[i_b, j_b, k_b] = a[i_a, j_a, k_a]

                                if a[i_a, j_a, k_a] >= b[i_b, j_b, k_b]:
                                    b[i_b, j_b, k_b] = a[i_a, j_a, k_a]
                                    i_ref = i_a
                                    j_ref = j_a
                        
                        self.mask[i_ref, j_ref, k_ref] = 1

        return b


    def superSample(self, a, stride_x, stride_y):

        size_x = (int)((np.shape(a)[0]))
        size_y = (int)((np.shape(a)[1]))
        size_z = (int)(np.shape(a)[2])

        i_x = size_x*stride_x
        i_y = size_y*stride_y

        if self.padded == True:
            i_x += 1
            i_y += 1

        b = np.zeros(((int)(i_x), (int)(i_y), size_z))

        for k_a in range(size_z):
            for i_a in range(size_x):
                for j_a in range(size_y):
                    for i_s in range(stride_x):
                        for j_s in range(stride_y):
                    
                            i_b = i_a * stride_x + i_s
                            j_b = j_a * stride_y + j_s
                            k_b = k_a                        

                            if i_b >= np.shape(b)[0] or j_b >= np.shape(b)[1] or i_b >= np.shape(self.mask)[0] or j_b >= np.shape(self.mask)[1]:
                                   continue

                            if self.mask[i_b, j_b, k_b] == 1:
                                b[i_b, j_b, k_b]  = a[i_a, j_a, k_a]

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



