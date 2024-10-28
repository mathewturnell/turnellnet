# TurnellNET

TurnellNET is a python naive implementation of a conventional Fully Connected (FC) or Convolutional (CNN) Neural Network.
The only libraries used for this implementation are numpy and matplotlib. Tensorflow is also used but only for loading the MINST data.
A Conda environment provided in conda.yml for convenience.

NN selection done with variable netType
Number of layers arbitrary and chosen by appending individual layers subclasses: FC (FullyConnected), CNN (Convolutional Layer), Slayer (Sampling Layer)
Default test_fc_minst.py includes examples of layer and hyperparameters definitions (Kernel numbers, Kernel sizes, sampling ratios, learning coefficients)
