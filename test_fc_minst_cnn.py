from fcnetwork import FCNetwork
from fclayer import FCLayer
from cnnlayer import CNNLayer
from activationlayer import ActivationLayer
from activation_functions import act_RELU, df_act_RELU
import tensorflow
from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
from matplotlib import pyplot as plt

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
#x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
#x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


net = FCNetwork()

net.add(CNNLayer(28, 28, 10, 10))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(act_RELU, df_act_RELU, False))
net.add(CNNLayer(19, 19, 2, 2))
net.add(ActivationLayer(act_RELU, df_act_RELU, True))
net.add(FCLayer(225, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(act_RELU, df_act_RELU, False))

net.train(x_train[0:3000], y_train[0:3000], iterations =10, learning_rate=0.1)

numberOfSamples = 10

out = net.predict(x_test[0:numberOfSamples])



images = np.reshape(x_test[0:numberOfSamples],(numberOfSamples,28,28))


fig, axs = plt.subplots(1,numberOfSamples)

for i in range(numberOfSamples):
    axs[i].imshow(images[i,:,:], interpolation='nearest')
    max = np.argmax(out[i])
    axs[i].title.set_text('%d'%(max))
    axs[i].axes.get_xaxis().set_visible(False)
    axs[i].axes.get_yaxis().set_visible(False)

fig1, axs1 = plt.subplots(10,10)

#w = net.input_layer.w.reshape(28,28,100)

#x = np.linspace(0, 27, 28)
#y = np.linspace(0, 27, 28)
#X, Y = np.meshgrid(x, y)

#for i in range(10):
#    for j in range(10):
#        c = axs1[i,j].pcolor(X, Y, w[:,:,i+j])
#        axs1[i,j].axes.get_xaxis().set_visible(False)
#        axs1[i,j].axes.get_yaxis().set_visible(False)

#fig1.colorbar(c, ax=axs1)

predictions = np.reshape(np.asarray(net.predictions), (np.size(net.predictions)))
J = np.reshape(net.J,(np.size(net.J)))

plt.figure(3)
plt.plot(range(len(J)), J)

plt.show()

#print("\n")
#print("predicted values : ")
#print(out, end="\n")
#print("true values : ")
#print(y_test[0:numberOfSamples])

