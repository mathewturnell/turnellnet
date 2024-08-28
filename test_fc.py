import numpy as np
from matplotlib import pyplot as plt
from fcnetwork import FCNetwork
from fclayer import FCLayer
from activationlayer import ActivationLayer

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = FCNetwork()
net.add(FCLayer(2,3))
net.add(ActivationLayer(3,3))
net.add(FCLayer(3,1))
net.add(ActivationLayer(1,1))

net.train(x_train, y_train, iterations = 1000, learning_rate = 0.1)

out = net.predict(x_train)
print(out)

predictions = np.reshape(np.asarray(net.predictions), (np.size(net.predictions)))
J = np.reshape(net.J,(np.size(net.J)))

fig, axs = plt.subplots(2)

axs[0].plot(range(len(predictions)), predictions)

axs[1].plot(range(len(predictions)), J)

plt.show()