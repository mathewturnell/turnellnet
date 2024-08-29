import numpy as np
from layer import Layer


class ActivationLayer(Layer):

    def __init__(self, f, d_f,output_reshape):

        self.f_output = f
        self.df_output = d_f
        self.output_reshape = output_reshape

    def forward_propagation(self, input_data, diag):

        self.x = input_data

        if(self.output_reshape == True):
            self.y = np.reshape(self.f_output(self.x), np.size(self.x))
            self.df = np.reshape(self.df_output(self.x),np.size(self.x))
        else:
            self.y = self.f_output(self.x)
            self.df = self.df_output(self.x)

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






