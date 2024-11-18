#(c) 2024 M. Turnell
#This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np

def act_tanh(x, N):

    out = x.copy()

    if(np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
            out[i] = np.tanh(x[i])
    elif (np.ndim(x) == 2):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                    out[i,j] = np.tanh(x[i,j])
    elif (np.ndim(x) == 3):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                for k in range(np.shape(x)[2]):
                    out[i,j,k] = np.tanh(x[i,j,k])

    return out


def df_act_tanh(x, N):

    out = x.copy()

    if(np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
            out[i] = 1 - np.tanh(x[i]) ** 2
    elif (np.ndim(x) == 2):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                    out[i,j] = 1 - np.tanh(x[i,j]) ** 2
    elif (np.ndim(x) == 3):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                for k in range(np.shape(x)[2]):
                    out[i,j,k] = 1 - np.tanh(x[i,j,k]) ** 2

    return out

def act_RELU(x, N):

    out = x.copy()

    if(np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
                if (x[i] >= 0):
                    out[i] = x[i]
                else:
                    out[i] = 0.01*x[i]
    elif(np.ndim(x) == 2):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                if (x[i,j] >= 0):
                    out[i,j] = x[i,j]
                else:
                    out[i,j] = 0.01*x[i,j]
    elif(np.ndim(x) == 3):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                for k in range(np.shape(x)[2]):
                    if(x[i,j,k]>=0):
                        out[i,j,k] = x[i,j,k]
                    else:
                        out[i,j,k] = 0.01*x[i,j,k]

    return out

def df_act_RELU(x, N):

    out = x.copy()

    if (np.ndim(x) == 1):
        for i in range(np.shape(x)[0]):
            if (x[i] > 0):
                out[i] = 1
            else:
                out[i] = 0.01
    elif (np.ndim(x) == 2):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                if (x[i, j] > 0):
                    out[i, j] = 1
                else:
                    out[i, j] = 0.01
    elif (np.ndim(x) == 3):
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                for k in range(np.shape(x)[2]):
                    if (x[i, j, k] > 0):
                        out[i, j, k] = 1
                    else:
                        out[i, j, k] = 0.01

    return out

def act_sigmoid(x, N):
    out = x.copy()
    return 1 / (1 + np.exp(-out))

def df_act_sigmoid(x, N):
    out = x.copy()
    s = act_sigmoid(out, N)
    return s * (1 - s)

def act_softmax(x, N):
    out = x.copy()
    out = np.exp(out)
    return out / np.sum(out)

def act_softmax_i(x, i, N):
    out = x.copy()
    tmp = np.exp(out)
    return tmp[0,i] / np.sum(tmp)

def df_act_softmax(x, N):
    
    out = x.copy()

    size = np.size(x)
    sm = np.zeros(size)

    for i in range(size):
        tmp = act_softmax_i(out, i, 0)
        sm[i] = tmp*(1-tmp)

    return sm