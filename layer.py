#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None

        self.prev_layer = None
        self.next_layer = None

        self.input_size = input_size
        self.output_size = output_size

    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass