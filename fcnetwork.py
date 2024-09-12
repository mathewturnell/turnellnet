# import numpy as np
import cupy as np

from datetime import datetime
import matplotlib.pyplot as plt

import error as er

import os

class FCNetwork:
    def __init__(self):

        self.layers = []
        self.output_layer = None
        self.input_layer = None
        self.expected_output = None

        # self.predictions = np.empty()
        # self.J = np.empty()

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

    def train(self, inputs_train, outputs_train, iterations, trainingSamples, diag):

        samples = len(inputs_train)

        for k in range(iterations):
            for i in range(samples):
                output = inputs_train[i].copy()

                for layer in self.layers:

                    start = datetime.now()

                    output = layer.forward_propagation(output,diag)

                    now = datetime.now()
                    print(now - start)
                    pass

                    
                    # index = layer.printState()
                    # if index == 1:
                    #     print('Real Value: %d, Predicted Value: %d'%(np.argmax(outputs_train[i+1]), np.argmax(output)))
                        



                error = er.error(outputs_train[i], output)
                d_error = er.d_error(error)

                delta_1 = d_error



                for layer in reversed(self.layers):
                    start = datetime.now()
                    delta_1 = layer.backward_propagation(delta_1, diag)
                    now = datetime.now()
                    print(now - start)
                    pass

               

                J_mse = er.J(error)

                self.printStateLearning(output, J_mse, k, i, np.argmax(outputs_train[i]) ,iterations, trainingSamples)

                now = datetime.now()
                print(now - start)


                #print('prediction %f, output %d' % (output, outputs_train[i]))
                # os.system('cls' if os.name == 'nt' else 'clear')
                # print('................................')
                # print('iteration %d/%d   J =%f' % (k + 1, iterations, J_mse))
                # print(output)
                # print(outputs_train[i])


                # self.predictions.append(output)
                # self.J.append(J_mse)
        
    def printStateLearning(self, output, error, iteration, sample, sampleTrueValue, iterations, trainingSamples):

        os.system('cls' if os.name == 'nt' else 'clear')

        a = "  _____                      _ _ _   _      _            ___  " + \
            "\n |_   _|   _ _ __ _ __   ___| | | \\ | | ___| |_  __   __/ _ \\ " + \
            "\n   | || | | | '__| '_ \\ / _ \\ | |  \\| |/ _ \\ __| \\ \\ / / | | |" + \
            "\n   | || |_| | |  | | | |  __/ | | |\\  |  __/ |_   \\ V /| |_| |" + \
            "\n   |_| \\__,_|_|  |_| |_|\\___|_|_|_| \\_|\\___|\\__|   \\_/  \\___/"
        
        

        print(a)
        print('\n')

        # print('Training NeuralNet, %d layers'%(np.size(self.layers)))
        print(self.printStructure())
        
        percentage = (iteration*trainingSamples+sample)*100/(trainingSamples*iterations)

        print('Sample: %d/%d, Iteration %d/%d'%(sample+1, trainingSamples, iteration+1, iterations))
        print('%d%% Complete'%(percentage))
        print(self.printPercentageBar(percentage))
        print('Real/Predicted Value: %d/%d'%(sampleTrueValue , np.argmax(output)))
        print('Error Energy: %.2f'%(error))

    def printStateLayers(self):
        for layer in self.layers:
            layer.printState()
    
    def printPercentageBar(self, percentage):

        s = ""
        digits = (int)(percentage / 10)
        for i in range(digits):
            s = s + "*"
        for i in range(digits, 10):
            s = s + "."
        
        return s
    
    def printStructure(self):

        s = "\n"
        for layer in self.layers:
            s = s + layer.type + (" (%dx%dx%d)->(%dx%dx%d)")%(layer.input_size_x,layer.input_size_y, layer.input_depth, layer.sizeOutput_x,layer.sizeOutput_y, layer.kernel_count) + "\n"
        
        return s

    def printEnergy(self):

        predictions = np.reshape(np.asarray(self.predictions), (np.size(self.predictions)))
        J = np.reshape(self.J,(np.size(self.J)))

        plt.figure()
        plt.plot(range(len(J)), J)


