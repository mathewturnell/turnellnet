import numpy as np
from layer import Layer


class ActivationLayer(Layer):

    def __init__(self, f, d_f, type, output_reshape):

        self.type = type

        self.f_output = f
        self.df_output = d_f
        self.output_reshape = output_reshape

        self.input_size_x = 0
        self.input_size_y = 0
        self.sizeOutput_x = 0
        self.sizeOutput_y = 0
        self.input_depth = 1
        self.kernel_count = 1

    def forward_propagation(self, input_data, diag):

        self.x = input_data

        if(self.output_reshape == True):
            self.y = np.reshape(self.f_output(self.x), (1, np.size(self.x)))
            self.df = np.reshape(self.df_output(self.x),(1, np.size(self.x)))

        else:
            self.y = self.f_output(self.x)
            self.df = self.df_output(self.x)

        self.input_size_x = np.size(self.x, 0)
        self.input_size_y = np.size(self.x, 1)
        self.sizeOutput_x = np.size(self.y, 0)
        self.sizeOutput_y = np.size(self.y, 1)

        self.input_depth = self.prev_layer.kernel_count
        self.kernel_count = 1

        return self.y

    def backward_propagation(self, output_error, diag):

        #if(np.ndim(output_error)>1):
        #    output_error = output_error[0:np.shape(self.df)[0],0:np.shape(self.df)[1]]
        d_1 = np.multiply(output_error, self.df)

        if(self.output_reshape == True):
            self.delta_1 = np.reshape(d_1, np.shape(self.x))
        else:
            self.delta_1 = d_1

        return self.delta_1
    
    def printState(self):

        return 0






