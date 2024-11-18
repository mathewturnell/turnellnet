#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from layer import Layer


class ActivationLayer(Layer):

    def __init__(self, f, d_f, N, type, output_reshape):

        self.type = type

        self.f_output = f
        self.df_output = d_f
        self.N = N
        self.output_reshape = output_reshape

        self.input_size_x = 0
        self.input_size_y = 0
        self.sizeOutput_x = 0
        self.sizeOutput_y = 0
        self.input_depth = 1
        self.kernel_count = 1
        self.delta = 0

        self.w = 0
        self.dw = 0

    def forward_propagation(self, input_data):

        self.x = np.copy(input_data)

        if(self.output_reshape == True):
            self.y = np.reshape(self.f_output(self.x, self.N), (1, np.size(self.x)))
            self.df = np.reshape(self.df_output(self.x, self.N),(1, np.size(self.x)))
        else:
            self.y = self.f_output(self.x, self.N)
            self.df = self.df_output(self.x, self.N)

        self.input_size_x = np.size(self.x, 0)
        self.input_size_y = np.size(self.x, 1)
        self.sizeOutput_x = np.size(self.y, 0)
        self.sizeOutput_y = np.size(self.y, 1)

        self.input_depth = self.prev_layer.kernel_count
        # self.kernel_count = np.size(self.x, 2)

        return self.y

    def backward_propagation(self, output_error, learning_rate):

        self.delta = np.copy(output_error)

        d_1 = np.multiply(output_error, self.df)

        if(self.output_reshape == True):
            self.delta_1 = np.reshape(d_1, np.shape(self.x))
        else:
            self.delta_1 = d_1

        return self.delta_1
    
    def printState(self, axs1):

        for k in range(self.kernel_count):

            if k >= 10:
                break

            axs1[k,0].xaxis.set_tick_params(labelbottom=False)
            axs1[k,1].xaxis.set_tick_params(labelleft=False)



            if self.x.ndim > 2:
                axs1[k,0].imshow(np.squeeze(self.x[:,:,k]), interpolation='nearest')
            else:
                axs1[k,0].plot(np.squeeze(self.x))

            if self.y.ndim > 2:
                axs1[k,1].imshow(np.squeeze(self.y[:,:,k]), interpolation='nearest')
            else:
                axs1[k,1].plot(np.squeeze(self.y))


            axs1[k,0].title.set_text('x[d=%d]'%(k))
            axs1[k,1].title.set_text('y[k=%d]'%(k))







