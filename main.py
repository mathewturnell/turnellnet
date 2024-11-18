#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

################################################################################################################

#Choice of NN: 0 = Fully Connected / 1 = Convolutional
netType = 1
#Choice of Dataset: 0 = MNIST digits / 1 = MNISTE letters
option = 1

import sys

from mnist import MnistDataloader
from fcnetwork import FCNetwork
from fclayer import FCLayer
from cnnlayer import CNNLayer
from slayer import Slayer
from activationlayer import ActivationLayer
from maxpool import Maxpool
from activation_functions import act_tanh, df_act_tanh, act_RELU, df_act_RELU, act_sigmoid, df_act_sigmoid, act_softmax, df_act_softmax

from tensorflow import keras
from keras._tf_keras.keras import utils as np_utils

import numpy as np
import string
from matplotlib import pyplot as plt

################################################################################################################

#Helper to load the 0-9 digit MNIST datatset
# def load_mnist():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     return (x_train, y_train), (x_test, y_test)

#Helper to load the Alphabet/Letter MNISTE expanded datatset. It's contained in the dataset folder.
def load_mnist(option):

    if option == 1:
        path_mnist_train_images = "./datasets/mniste/emnist-letters-train-images-idx3-ubyte"
        path_mnist_train_labels = "./datasets/mniste/emnist-letters-train-labels-idx1-ubyte"
        path_mnist_test_images = "./datasets/mniste/emnist-letters-train-images-idx3-ubyte"
        path_mnist_test_labels = "./datasets/mniste/emnist-letters-train-labels-idx1-ubyte"
    else:
        path_mnist_train_images = "./datasets/mnist/train-images.idx3-ubyte"
        path_mnist_train_labels = "./datasets/mnist/train-labels.idx1-ubyte"
        path_mnist_test_images = "./datasets/mnist/t10k-images.idx3-ubyte"
        path_mnist_test_labels = "./datasets/mnist/t10k-labels.idx1-ubyte"

    mnist_dataloader = MnistDataloader(path_mnist_train_images, path_mnist_train_labels, path_mnist_test_images, path_mnist_test_labels)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    return (x_train, y_train), (x_test, y_test)




################################################################################################################
#Hyperparameters

#Input Parameters

inputDepth = 1
inputSize_x = 28
inputSize_y = 28

#hyper parameters CNN

numberOfKernels0 = 16
numberOfKernels1 = 24
numberOfKernels2 = 24

kernelSize0_x = 5
kernelSize0_y = 5
kernelSize1_x = 3
kernelSize1_y = 3
kernelSize2_x = 3
kernelSize2_y = 3

trainingSamples = 100000
trainingIterations = 1
targetAcc = 99 #target accuracy required to stop learning phase and move to prediction.

trainingRateCNN = 0.3
trainingRateFC = 0.3

numberOfClasses = 10

# load MNIST dataset
# (x_train, y_train), (x_test, y_test) = load_mnist()

# load MNISTE dataset
(x_train, y_train), (x_test, y_test) = load_mnist(option)

# reshape and normalize input data
if(netType == 0):
    x_train = np.reshape(x_train, (np.shape(x_train)[0], 1, 28*28))
else:
    x_train = np.expand_dims(x_train, axis = 3)

x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)

# If a FC network is chosen, the 28x28 image 2-D array needs to be flattened to a 1-D 784 array.
if(netType == 0):
    x_test = np.reshape(x_test, (np.shape(x_test)[0], 1, 28*28))
else:
    x_test = np.expand_dims(x_test, axis = 3)

x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# This line is only used if using the MNISTE dataset, image x and y axis seem to be swapped.
if option == 1:
    x_train = np.transpose(x_train, (0, 2, 1, 3))
    x_test = np.transpose(x_test, (0, 2, 1, 3))

################################################################################################################

if option == 0:
    numberOfClasses = 10
else:
    numberOfClasses = 27

#Declares new Neural Network object.
net = FCNetwork(trainingRateCNN)

if(netType == 0):
    net.add(FCLayer(28*28, 100, 0.1))
    net.add(ActivationLayer(act_RELU, df_act_RELU, 1, "Act_tanh", True))

    net.add(FCLayer(100, 50, trainingRateFC))
    net.add(ActivationLayer(act_tanh, df_act_tanh, 1, "act_tanh", 0))

    net.add(FCLayer(50, numberOfClasses, trainingRateFC)) 
    net.add(ActivationLayer(act_tanh, df_act_tanh, 1, "Act_tanh", 0))


else:

    #Instantiate FC, CNN and Downsampling layers.
    #In the example bellow the Network contains 2 CNN and FC layers.

    cnn0 = CNNLayer(inputSize_x, inputSize_y, inputDepth, kernelSize0_x, kernelSize0_y, numberOfKernels0, 1, trainingRateCNN)
    # ds0 = Slayer(cnn0.sizeOutput_x, cnn0.sizeOutput_y, cnn0.kernel_count, 2, 2, trainingRateCNN)
    mp0 = Maxpool(cnn0.sizeOutput_x, cnn0.sizeOutput_y, cnn0.kernel_count, 2, 2)

    # cnn1 = CNNLayer(ds0.sizeOutput_x, ds0.sizeOutput_y, ds0.kernel_count, kernelSize1_x, kernelSize1_y, numberOfKernels1, 1, trainingRateCNN)
    cnn1 = CNNLayer(mp0.sizeOutput_x, mp0.sizeOutput_y, mp0.kernel_count, kernelSize1_x, kernelSize1_y, numberOfKernels1, 1, trainingRateCNN)
    # ds1 = Slayer(cnn1.sizeOutput_x, cnn1.sizeOutput_y, cnn1.kernel_count, 2, 2, trainingRateCNN)
    mp1 = Maxpool(cnn1.sizeOutput_x, cnn1.sizeOutput_y, cnn1.kernel_count, 2, 2)

    # cnn2 = CNNLayer(mp1.sizeOutput_x, mp1.sizeOutput_y, mp1.kernel_count, kernelSize2_x, kernelSize2_y, numberOfKernels2, 1, trainingRateCNN)
    # ds2 = Slayer(cnn2.sizeOutput_x, cnn2.sizeOutput_y, cnn2.kernel_count, 2, 2, trainingRateCNN)
    # mp2 = Maxpool(cnn2.sizeOutput_x, cnn2.sizeOutput_y, cnn2.kernel_count, 2, 2)

    net.add(cnn0)
    net.add(mp0)
    net.add(ActivationLayer(act_RELU, df_act_RELU, 1, "act_RELU", False))

    net.add(cnn1)
    net.add(mp1)
    net.add(ActivationLayer(act_RELU, df_act_RELU, 1, "act_RELU", True))

    # net.add(cnn2)
    # net.add(mp2)
    # net.add(ActivationLayer(act_RELU, df_act_RELU, 1, "act_RELU", True))

    # net.add(cnn2)
    # net.add(ds2)
    # net.add(ActivationLayer(act_RELU, df_act_RELU, 1, "Act_RELU", True))

    net.add(FCLayer(mp1.sizeOutput_x * mp1.sizeOutput_y * mp1.kernel_count, numberOfClasses, trainingRateFC))
    net.add(ActivationLayer(act_sigmoid, df_act_sigmoid, 1, "act_softmax", False))

# net.add(FCLayer(100, 50, trainingRateFC))
# net.add(ActivationLayer(act_tanh, df_act_tanh, 1, "act_tanh", 0))



# net.add(FCLayer(50, numberOfClasses, trainingRateFC)) 
# net.add(ActivationLayer(act_tanh, df_act_tanh, 1, "Act_tanh", 0))

#Start of learning with back propagation. Learning will continue until targetaccuracy is reached.
net.train(x_train[0:trainingSamples], y_train[0:trainingSamples], iterations =trainingIterations, trainingSamples=trainingSamples , targetAccuracy=targetAcc, diag=1)

#Number of test samples to evaluate results.
numberOfSamples = 50
out = net.predict(x_test[0:numberOfSamples], diag=1)
images = np.reshape(x_test[0:numberOfSamples],(numberOfSamples,28,28))

fig, axs = plt.subplots((int)(numberOfSamples/10), 10)

for i in range(numberOfSamples):

    x = (int)(i/10)
    y = i%10

    axs[x, y].imshow(images[i,:,:], interpolation='nearest')
    max = np.argmax(out[i])

    if option == 0:
        axs[x, y].title.set_text('%d'%(max))
    else:
        axs[x, y].title.set_text(string.ascii_lowercase[max-1])
    axs[x, y].axes.get_xaxis().set_visible(False)
    axs[x, y].axes.get_yaxis().set_visible(False)

plt.show()
