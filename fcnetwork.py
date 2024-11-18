#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
import matplotlib.pyplot as plt
import error as er

import os

class FCNetwork:
    def __init__(self, learningRate):

        self.layers = []
        self.output_layer = None
        self.input_layer = None
        self.expected_output = None


        self.predictions = []
        self.accuracy = []
        self.J = []

        self.rollingResults = np.zeros(200)
        self.rollingIndex = 0
        self.maxAccuracy = 0

        self.trainingIteration = 0
        self.learningRate = learningRate

        self.plotFigs = []
        self.plotAxis = []

    def add(self, layer):

        self.layers.append(layer)
        self.output_layer = layer

        if(len(self.layers) > 1):
            self.layers[len(self.layers)-2].next_layer = layer
            layer.prev_layer = self.layers[len(self.layers) - 2]
        else:
            self.input_layer = self.layers[len(self.layers) - 2]

    def predict(self, input_data, diag):

        limitGraphsX = 10
        limitGraphsY = 10

        numberLinesOutput = 10

        samples = len(input_data)
        result = []

        for i in range(samples):

            output = input_data[i].copy()

            for layer in self.layers:
                    output = layer.forward_propagation(output)          

            result.append(output.copy())

        self.initializePlots()

        return result

    def train(self, inputs_train, outputs_train, iterations, trainingSamples, targetAccuracy, diag):

        samples = len(inputs_train)

        counter = 0

        for k in range(iterations):
            for i in range(samples):
                output = inputs_train[i].copy()

                for layer in self.layers:
                    output = layer.forward_propagation(output)
                        
                prediction = np.argmax(output)
                groundTruth = np.argmax(outputs_train[i])

                if(prediction == groundTruth):
                    self.rollingResults[self.rollingIndex] = 1
                else:
                    self.rollingResults[self.rollingIndex] = 0

                self.rollingIndex = (self.rollingIndex + 1)%200

                error = er.error(outputs_train[i], output)
                d_error = er.d_error(error)

                learningRate = self.getLearningRate()

                delta_1 = d_error
                for layer in reversed(self.layers):
                    delta_1 = layer.backward_propagation(delta_1, learningRate)

                J_mse = er.J(error)
                rollingAccuracy = self.getAccuracy()

                if rollingAccuracy > self.maxAccuracy:
                    self.maxAccuracy = rollingAccuracy

                # self.J.append(J_mse)
                # self.accuracy.append(rollingAccuracy)

                self.printStateLearning(prediction, J_mse, rollingAccuracy, k, i, groundTruth ,iterations, trainingSamples)

                self.trainingIteration += 1

                counter += 1

                if counter%25000 == 0 :
                    self.initializePlots()

                if rollingAccuracy >= targetAccuracy:
                    break
            
        # self.printEnergy()
        # self.initializePlots()

    def initializePlots(self):

        for i,layer in enumerate(self.layers):

            # if layer.sizeOutput_y == 1:
            #     continue

            numberLinesOutput = layer.kernel_count
            
            if numberLinesOutput < 2:
                numberLinesOutput = 2
            if numberLinesOutput > 10:
                numberLinesOutput = 10

            fig1, axs1 = plt.subplots(numberLinesOutput,3)
            fig1.suptitle("Layer %d %s %dx%dx%d -> %dx%dx%d" % (i, layer.type, layer.input_size_x, layer.input_size_y, layer.input_depth, layer.sizeOutput_x, layer.sizeOutput_y, layer.kernel_count))
            layer.printState(axs1)

    def getLearningRate(self):
            
        learningRate = self.learningRate* np.exp(-self.trainingIteration/200000)
        if learningRate < 0.01:
            learningRate = 0.01
        
        return learningRate
        
    def printStateLearning(self, output, error, accuracy, iteration, sample, sampleTrueValue, iterations, trainingSamples):

        os.system('cls' if os.name == 'nt' else 'clear')

        a = "  _____                      _ _ _   _      _            ___  " + \
            "\n |_   _|   _ _ __ _ __   ___| | | \\ | | ___| |_  __   __/ _ \\ " + \
            "\n   | || | | | '__| '_ \\ / _ \\ | |  \\| |/ _ \\ __| \\ \\ / / | | |" + \
            "\n   | || |_| | |  | | | |  __/ | | |\\  |  __/ |_   \\ V /| |_| |" + \
            "\n   |_| \\__,_|_|  |_| |_|\\___|_|_|_| \\_|\\___|\\__|   \\_/  \\___/"
        
        

        print(a)
        print('\n')

        print('Training NeuralNet, %d layers'%(np.size(self.layers)))
        print(self.printStructure())
        
        percentage = (iteration*trainingSamples+sample)*100/(trainingSamples*iterations)

        print('Sample: %d/%d, Iteration %d/%d'%(sample+1, trainingSamples, iteration+1, iterations))
        print('%d%% Complete'%(percentage))
        print(self.printPercentageBar(percentage))
        print('Real/Predicted Value: %d/%d'%(sampleTrueValue , output))

        print('Rolling Accuracy: %d%%/%d%% '%(accuracy, self.maxAccuracy))
        print('Error Energy: %.6f'%(error))
        print('Learning Rate: %.6f'%(self.getLearningRate()))

        for i, layer in enumerate(self.layers):
            print('Layer %d' % i)
            print('Mean W: %.6f' % np.average(layer.w))
            print('Mean dW: %.6f' % np.average(layer.dw))
            print('Mean delta: %.6f' % np.average(layer.delta))
        

    
    def printPercentageBar(self, percentage):

        s = ""
        digits = (int)(percentage / 10)
        for i in range(digits):
            s = s + "*"
        for i in range(digits, 10):
            s = s + "."
        
        return s
    
    def getAccuracy(self):
        return (float)(100*np.count_nonzero(self.rollingResults))/(len(self.rollingResults))
    
    def printStructure(self):

        s = "\n"
        for layer in self.layers:
            s = s + layer.type + (" (%dx%dx%d)->(%dx%dx%d)")%(layer.input_size_x,layer.input_size_y, layer.input_depth, layer.sizeOutput_x,layer.sizeOutput_y, layer.kernel_count) + "\n"
        
        return s

    def printEnergy(self):

        J = np.reshape(self.J,(np.size(self.J)))
        accuracy= np.reshape(self.accuracy,(np.size(self.accuracy)))

        fig1, axs1 = plt.subplots(2,1)
        axs1[0].plot(range(len(J)), J)
        axs1[1].plot(range(len(accuracy)), accuracy)

        axs1[0].title.set_text('Loss Function J (mse)')
        axs1[1].title.set_text('Accuracy (%)')


