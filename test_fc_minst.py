from fcnetwork import FCNetwork
from fclayer import FCLayer
from cnnlayer import CNNLayer
from activationlayer import ActivationLayer
from activation_functions import act_tanh, df_act_tanh

from tensorflow import keras
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras import utils as np_utils

import numpy as np
from matplotlib import pyplot as plt

################################################################################################################


#Choice of NN

netType = 1

#Input Parameters

inputDepth = 1

inputSize_x = 28
inputSize_y = 28

#hyper parameters CNN

numberOfKernels = 10
numberOfKernels1 = 10
numberOfKernels2 = 10

kernelSize_x = 5
kernelSize_y = 5
kernelSize1_x = 15
kernelSize1_y = 15

kernelSize2_x = 10
kernelSize2_y = 10

trainingSamples = 5000
trainingIterations = 2

trainingRateCNN = 0.1
trainingRateFC = 0.1


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalize input data
if(netType == 0):
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
else:
    x_train = np.expand_dims(x_train, axis = 3)

x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)


if(netType == 0):
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
else:
    x_test = np.expand_dims(x_test, axis = 3)

x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

################################################################################################################

net = FCNetwork()

if(netType == 0):
    net.add(FCLayer(28*28, 2700, 0.1))
    net.add(ActivationLayer(act_tanh, df_act_tanh, 1))
    net.add(FCLayer(2700, 100, 0.1))
else:

    cnn0 = CNNLayer(inputSize_x, inputSize_y, inputDepth, kernelSize_x, kernelSize_y, numberOfKernels, trainingRateCNN)
    cnn1 = CNNLayer(cnn0.sizeOutput_x, cnn0.sizeOutput_y, cnn0.kernel_count, kernelSize1_x, kernelSize1_y, numberOfKernels1, trainingRateCNN)
    #cnn2 = CNNLayer(cnn1.sizeOutput_x, cnn1.sizeOutput_y, cnn1.kernel_count, kernelSize2_x, kernelSize2_y, numberOfKernels2, trainingRateCNN)

    net.add(cnn0)
    net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", False))
    net.add(cnn1)
    net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", True))
    # net.add(cnn2)
    # net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", True))
    net.add(FCLayer(cnn1.sizeOutput_x * cnn1.sizeOutput_y *numberOfKernels1, 100, trainingRateFC))

net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", 0))
net.add(FCLayer(100, 50, 0.1))
net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", 0))
net.add(FCLayer(50, 10, 0.1)) 
net.add(ActivationLayer(act_tanh, df_act_tanh, "Act_tanh", 0))

net.train(x_train[0:trainingSamples], y_train[0:trainingSamples], iterations =trainingIterations, trainingSamples=trainingSamples ,diag=1)

numberOfSamples = 10
out = net.predict(x_test[0:numberOfSamples], diag=1)
images = np.reshape(x_test[0:numberOfSamples],(numberOfSamples,28,28))

fig, axs = plt.subplots(1,numberOfSamples)

for i in range(numberOfSamples):
    axs[i].imshow(images[i,:,:], interpolation='nearest')
    max = np.argmax(out[i])
    axs[i].title.set_text('%d'%(max))
    axs[i].axes.get_xaxis().set_visible(False)
    axs[i].axes.get_yaxis().set_visible(False)

net.printStateLayers()
net.printEnergy()

plt.show()

pass



