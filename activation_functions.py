import numpy as np

def act_tanh(x):

    out = x.copy()
    for i in range(len(x)):
        out[i] = np.tanh(x[i])

    return out


def df_act_tanh(x):

    out = x.copy()
    for i in range(len(x)):
        out[i] = 1 - np.tanh(x[i]) ** 2

    return out

def act_RELU(x):

    out = x.copy()

    if(np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
                if (x[i] >= 0):
                    out[i] = x[i]
                else:
                    out[i] = 0
    else:
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                if(x[i,j]>=0):
                    out[i,j] = x[i,j]
                else:
                    out[i,j] = 0

    return out

def df_act_RELU(x):

    out = x.copy()

    if(np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
                if (x[i] >= 0):
                    out[i] = 1
                else:
                    out[i] = 0
    else:
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                if(x[i,j]>=0):
                    out[i,j] = 1
                else:
                    out[i,j] = 0

    return out