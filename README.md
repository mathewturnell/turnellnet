#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

TurnellNET is a python naive implementation of a conventional Fully Connected (FC) and/or Convolutional (CNN) Neural Network intented for learning and experimentation. 
Python classes are used to fully implement FC or CNNs with an aribitrary number of Fully Connected, Convultional, Activation and Downsampling Layers.

Implementation is naive due to intentional use of index based operations instead of using tensor based APIs like PyTorch or TensorFlow that allow for GPU optimizations (ie: CUDA)
The index notation helps understand the implementation of classic back-propagation method that happen during supervised learning.

The only libraries used for this implementation are numpy and matplotlib. 
Copy of MNIST datasets are provided in the datasets folder.
A Conda environment provided in conda.yml for convenience.



