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

numberOfKernels = 3

kernelSize_x = 3
kernelSize_y = 3

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
    net.add(CNNLayer(inputSize_x, inputSize_y, inputDepth, kernelSize_x, kernelSize_y, numberOfKernels, 0.1))
    net.add(ActivationLayer(act_tanh, df_act_tanh, True))
    net.add(FCLayer((inputSize_x + kernelSize_x - 1) * (inputSize_y + kernelSize_y - 1)*numberOfKernels, 100, 0.1))

net.add(ActivationLayer(act_tanh, df_act_tanh, 0))
net.add(FCLayer(100, 50, 0.1))
net.add(ActivationLayer(act_tanh, df_act_tanh, 0))
net.add(FCLayer(50, 10, 0.1)) 
net.add(ActivationLayer(act_tanh, df_act_tanh, 0))

net.train(x_train[0:10], y_train[0:10], iterations =10, diag=1)

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

for layer in net.layers:
    layer.printState()

plt.show()

pass

# w = net.input_layer.w.reshape(28,28,100)

# x = np.linspace(0, 27, 28)
# y = np.linspace(0, 27, 28)
# X, Y = np.meshgrid(x, y)

# for i in range(10):
#     for j in range(10):
#         c = axs1[i,j].pcolor(X, Y, w[:,:,i+j])
#         axs1[i,j].axes.get_xaxis().set_visible(False)
#         axs1[i,j].axes.get_yaxis().set_visible(False)

# fig1.colorbar(c, ax=axs1)

predictions = np.reshape(np.asarray(net.predictions), (np.size(net.predictions)))
J = np.reshape(net.J,(np.size(net.J)))

# plt.figure(3)
# plt.plot(range(len(J)), J)

#print("\n")
#print("predicted values : ")
#print(out, end="\n")
#print("true values : ")
#print(y_test[0:numberOfSamples])

