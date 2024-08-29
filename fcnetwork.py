import numpy as np
import matplotlib.pyplot as plt
import error as er

import os

class FCNetwork:
    def __init__(self):

        self.layers = []
        self.output_layer = None
        self.input_layer = None
        self.expected_output = None

        self.predictions = []
        self.J = []

    def add(self, layer):

        self.layers.append(layer)
        self.output_layer = layer

        if(len(self.layers) > 1):
            self.layers[len(self.layers)-2].next_layer = layer
            layer.prev_layer = self.layers[len(self.layers) - 2]
        else:
            self.input_layer = self.layers[len(self.layers) - 2]

    def predict(self, input_data, diag):

        samples = len(input_data)
        result = []



        for i in range(samples):
            output = input_data[i].copy()

            z = 0

            for layer in self.layers:
                output = layer.forward_propagation(output, diag)

                # if z == 0:
                #     os.system('cls' if os.name == 'nt' else 'clear')
                #     print('Input (Layer z=%d)'%z)
                #     print(input_data[i].copy())
                #     print('Feature Map (Layer z=%d)'%z)
                #     print(output)
                #     layer.printState()
                #     pass

                
                z = z + 1

            # os.system('cls' if os.name == 'nt' else 'clear')
            # print('Predicted Value: %d'%(np.argmax(output)))


            result.append(output.copy())

        return result

    def train(self, inputs_train, outputs_train, iterations, diag):

        samples = len(inputs_train)

        for k in range(iterations):
            for i in range(samples):
                output = inputs_train[i].copy()

                print('Real Value: %d'%(np.argmax(outputs_train[i])))

                for layer in self.layers:
                    output = layer.forward_propagation(output,diag)
                    
                    # index = layer.printState()
                    # if index == 1:
                    #     print('Real Value: %d, Predicted Value: %d'%(np.argmax(outputs_train[i+1]), np.argmax(output)))
                        

                error = er.error(outputs_train[i], output)
                d_error = er.d_error(error)

                delta_1 = d_error
                for layer in reversed(self.layers):
                    delta_1 = layer.backward_propagation(delta_1, diag)

                J_mse = er.J(error)

                print('Predicted Value: %d'%(np.argmax(output)))

                #print('prediction %f, output %d' % (output, outputs_train[i]))
                # os.system('cls' if os.name == 'nt' else 'clear')
                # print('................................')
                # print('iteration %d/%d   J =%f' % (k + 1, iterations, J_mse))
                # print(output)
                # print(outputs_train[i])


                self.predictions.append(output)
                self.J.append(J_mse)


